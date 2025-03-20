#include "slict/FeatureCloud.h"

#include "STDesc.h"
#include "utility.h"

#include "PoseLocalParameterization.h"
#include <ceres/ceres.h>

// UFO
#include <ufo/map/code/code_unordered_map.h>
#include <ufo/map/point_cloud.h>
#include <ufo/map/surfel_map.h>
#include <ufomap_msgs/UFOMapStamped.h>
#include <ufomap_msgs/conversions.h>
#include <ufomap_ros/conversions.h>

/* All needed for filter of custom point type----------*/
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/impl/crop_box.hpp>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/filters/impl/uniform_sampling.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/pcl_base.h>

#include "GaussianProcess.hpp"
// Shorthands
typedef sensor_msgs::PointCloud2::ConstPtr rosCloudMsgPtr;
typedef sensor_msgs::PointCloud2 rosCloudMsg;
typedef Sophus::SO3d SO3d;

// Shorthands for ufomap
namespace ufopred = ufo::map::predicate;
using ufoSurfelMap = ufo::map::SurfelMap;
using ufoSurfelMapPtr = boost::shared_ptr<ufoSurfelMap>;
using ufoNode = ufo::map::NodeBV;
using ufoSphere = ufo::geometry::Sphere;
using ufoPoint3 = ufo::map::Point3;

class LITELOAM
{

private:
    ros::NodeHandlePtr nh_ptr;

    // Subscriber
    ros::Subscriber lidarCloudSub;

    // Publisher
    ros::Publisher relocPub;

    // Queue + mutex
    std::deque<CloudXYZIPtr> cloud_queue;
    std::mutex buffer_mutex;

    // Clouds on the sliding window
    deque<CloudXYZIPtr> SwCloud;

    // Prior map + kd-tree
    CloudXYZIPtr priorMap;
    KdFLANNPtr kdTreeMap;

    // Thread for processing
    std::thread processThread;
    bool running = true;

    // Initial timestamp
    double startTime;

    // Initial pose (only a single variable)
    mytf initPose;
    
    // The id of the loam instance
    int liteloam_id;

public:

    ~LITELOAM()
    {
        // Dừng vòng lặp trong processBuffer()
        running = false;

        // Đợi thread kết thúc
        if (processThread.joinable())
            processThread.join();

        ROS_WARN("liteloam %d destructed." RESET, liteloam_id);
    }

    LITELOAM(const CloudXYZIPtr &priorMap, const KdFLANNPtr &kdTreeMap,
             const mytf &initPose, int id, const ros::NodeHandlePtr &nh_ptr)
        : priorMap(priorMap), kdTreeMap(kdTreeMap),
          initPose(initPose), liteloam_id(id), nh_ptr(nh_ptr)
    {

        // sub to imu , optimization
        lidarCloudSub = nh_ptr->subscribe("/lastcloud", 100, &LITELOAM::PCHandler, this); // Change lastcloud_inB to lastcloud
        relocPub = nh_ptr->advertise<geometry_msgs::PoseStamped>("/liteloam_pose", 100);

        std::cout << "[LITELOAM] " << liteloam_id
                  << " Subscribed to /lastcloud and publishing to /liteloam_pose"
                  << std::endl;

        startTime = ros::Time::now().toSec();

        running = true;
        processThread = std::thread(&LITELOAM::processBuffer, this);

        ROS_INFO_STREAM("[LITELOAM " << liteloam_id << "] Constructor - thread started");
    }

    void stop() {running = false;}

    void processBuffer()
    {
      while (running && ros::ok())
      {
          if (cloud_queue.empty())
          {
              std::this_thread::sleep_for(std::chrono::milliseconds(10));
              continue;
          }
  
          // Start measuring processing time
          ros::Time startProcessingTime = ros::Time::now();
  
          CloudXYZIPtr cloudToProcess;
          {
              std::lock_guard<std::mutex> lock(buffer_mutex);
              cloudToProcess = cloud_queue.front();
              cloud_queue.pop_front();
          }
          
          uint64_t pcl_timestamp = cloudToProcess->header.stamp;
          ros::Time cloudTimestamp = ros::Time(pcl_timestamp / 1e6, (pcl_timestamp % static_cast<uint64_t>(1e6)) * 1e3);
  
          if (liteloam_id == 0)
              ROS_INFO("liteloam_id %d. Start time: %f. Running Time: %f.",
                       liteloam_id, startTime, timeSinceStart());
  
          //------------------------------------------------
          // 1) Pre-processing of input cloud
          //------------------------------------------------
          pcl::PointCloud<pcl::PointXYZI>::Ptr sourceCloud(new pcl::PointCloud<pcl::PointXYZI>());
          pcl::copyPointCloud(*cloudToProcess, *sourceCloud);
  
          // Apply VoxelGrid filter to downsample the point cloud
          pcl::VoxelGrid<pcl::PointXYZI> vg;
          vg.setInputCloud(sourceCloud);
          vg.setLeafSize(0.3f, 0.3f, 0.3f);  // Adjust accordingly
          pcl::PointCloud<pcl::PointXYZI>::Ptr sourceFiltered(new pcl::PointCloud<pcl::PointXYZI>());
          vg.filter(*sourceFiltered);
  
          //------------------------------------------------
          // 2) Configure ICP
          //------------------------------------------------
          pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
          icp.setInputSource(sourceFiltered);  // Source cloud (moving)
          icp.setInputTarget(priorMap);        // Target cloud (static reference map)
          
          // Set ICP parameters
          icp.setMaxCorrespondenceDistance(2.0);   // Max distance for point correspondence
          icp.setMaximumIterations(50);           // Max iterations for convergence
          icp.setTransformationEpsilon(1e-6);     // Stop if transformation difference < threshold
          icp.setEuclideanFitnessEpsilon(1e-6);   // Stop if fitness error < threshold
          
          // Sử dụng initPose làm initial guess
          Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
          initial_guess.block<3, 3>(0, 0) = initPose.rot.toRotationMatrix().cast<float>(); // Gán ma trận quay từ initPose
          initial_guess.block<3, 1>(0, 3) = initPose.pos.cast<float>(); // Gán vị trí từ initPose
  
          //------------------------------------------------
          // 3) Execute ICP
          //------------------------------------------------
          pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());
          icp.align(*aligned, initial_guess);
  
          // Retrieve ICP fitness score
          double icpFitnessThres = 0.3;   // Threshold for ICP result acceptance
          double icpFitnessRes = icp.getFitnessScore();
  
          //------------------------------------------------
          // 4) Evaluate ICP result
          //------------------------------------------------
          if (icp.hasConverged()) // && icpFitnessRes <= icpFitnessThres)
          {
              ROS_INFO_STREAM("ICP converged. Score: " << icpFitnessRes);
  
              // Get the final transformation matrix from ICP
              Eigen::Matrix4f icpTransformation = icp.getFinalTransformation();
              Eigen::Matrix3f R = icpTransformation.block<3,3>(0,0);
              Eigen::Vector3f t = icpTransformation.block<3,1>(0,3);
  
              Eigen::Quaternionf q(R);
              geometry_msgs::PoseStamped pose_msg;
  
              // Calculate processing duration
              ros::Duration processingDuration = ros::Time::now() - startProcessingTime;
  
              // Set pose message timestamp as the cloud's original timestamp + processing duration
              pose_msg.header.stamp = cloudTimestamp + processingDuration;
              pose_msg.header.frame_id = "map";
  
              // Assign position
              pose_msg.pose.position.x = t.x();
              pose_msg.pose.position.y = t.y();
              pose_msg.pose.position.z = t.z();
  
              // Assign orientation
              pose_msg.pose.orientation.x = q.x();
              pose_msg.pose.orientation.y = q.y();
              pose_msg.pose.orientation.z = q.z();
              pose_msg.pose.orientation.w = q.w();
  
              // Publish estimated pose
              relocPub.publish(pose_msg);
          }
          else
          {
              ROS_WARN("ICP did not converge or Fitness Score (%f) > Threshold (%f)", icpFitnessRes, icpFitnessThres);
          }
  
          //------------------------------------------------
          // 5) Repeat loop
          //------------------------------------------------
      }
  
      if (liteloam_id == 0)
          ROS_INFO(KRED "liteloam_id %d exits." RESET, liteloam_id);
  }

    void PCHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(buffer_mutex);

        // Chuyển đổi dữ liệu từ ROS PointCloud2 sang PCL
        CloudXYZIPtr newCloud(new CloudXYZI());
        pcl::fromROSMsg(*msg, *newCloud);

        // Copy dữ liệu vào hàng đợi với timestamp tương ứng
        cloud_queue.push_back(newCloud);
    }

    void Associate(const KdFLANNPtr &kdtreeMap,
                   const CloudXYZIPtr &priormap, const CloudXYZITPtr &cloudRaw,
                   const CloudXYZIPtr &cloudInB, const CloudXYZIPtr &cloudInW,
                   vector<LidarCoef> &Coef)
    {
        ROS_ASSERT_MSG(cloudRaw->size() == cloudInB->size(),
                       "cloudRaw: %d. cloudInB: %d", cloudRaw->size(),
                       cloudInB->size());
        
        int knnSize = 6;
        double minKnnSqDis = 0.5*0.5;
        double min_planarity = 0.2, max_plane_dis = 0.3;

        if (priormap->size() > knnSize)
        {
            int pointsCount = cloudInW->points.size();
            vector<LidarCoef> Coef_;
            Coef_.resize(pointsCount);

            #pragma omp parallel for num_threads(MAX_THREADS)
            for (int pidx = 0; pidx < pointsCount; pidx++)
            {
                double tpoint = cloudRaw->points[pidx].t;
                PointXYZIT pointRaw = cloudRaw->points[pidx];
                PointXYZI pointInB = cloudInB->points[pidx];
                PointXYZI pointInW = cloudInW->points[pidx];

                Coef_[pidx].n = Vector4d(0, 0, 0, 0);
                Coef_[pidx].t = -1;

                if (!Util::PointIsValid(pointInB))
                {
                    pointInB.x = 0;
                    pointInB.y = 0;
                    pointInB.z = 0;
                    pointInB.intensity = 0;
                    continue;
                }

                if (!Util::PointIsValid(pointInW))
                    continue;


                vector<int> knn_idx(knnSize, 0);
                vector<float> knn_sq_dis(knnSize, 0);
                kdtreeMap->nearestKSearch(pointInW, knnSize, knn_idx, knn_sq_dis);

                vector<PointXYZI> nbrPoints;
                if (knn_sq_dis.back() < minKnnSqDis)
                    for (auto &idx : knn_idx)
                        nbrPoints.push_back(priormap->points[idx]);
                else
                    continue;

                // Fit the plane
                if (Util::fitPlane(nbrPoints, min_planarity, max_plane_dis,
                                   Coef_[pidx].n, Coef_[pidx].plnrty))
                {
                    Coef_[pidx].t = tpoint;
                    Coef_[pidx].f = Vector3d(pointRaw.x, pointRaw.y, pointRaw.z);
                    Coef_[pidx].finW = Vector3d(pointInW.x, pointInW.y, pointInW.z);
                    Coef_[pidx].fdsk = Vector3d(pointInB.x, pointInB.y, pointInB.z);
                }
            }

            // Copy the coefficients to the buffer
            Coef.clear();
            int totalFeature = 0;
            for (int pidx = 0; pidx < pointsCount; pidx++)
            {
                LidarCoef &coef = Coef_[pidx];
                if (coef.t >= 0)
                {
                    Coef.push_back(coef);
                    Coef.back().ptIdx = totalFeature;
                    totalFeature++;
                }
            }
        }
    }

    bool hasMoved(const SE3d &currentPose) const
    {
        double pos_diff = (currentPose.translation() - initPose.pos).norm();
        return (pos_diff > 1);
    }

    // Getter for Relocalization
    double timeSinceStart() const { return ros::Time::now().toSec() - startTime; }
    bool loamConverged() const { return false; }
    bool isRunning() const { return running; }
    int getID() const { return liteloam_id; }
};

class Relocalization
{

private:
    // Node handler
    ros::NodeHandlePtr nh_ptr;
    
    // Subcriber of lidar pointcloud
    ros::Subscriber lidarCloudSub;

    // Subscriber to uloc prediction
    ros::Subscriber ulocSub;

    // Relocalization pose publication
    ros::Publisher relocPub;

    CloudXYZIPtr priorMap;
    KdFLANNPtr kdTreeMap;
    bool priorMapReady = false;

    mutex loam_mtx;
    std::vector<std::shared_ptr<LITELOAM>> loamInstances;
    std::thread checkLoamThread;

public:
    // Destructor
    ~Relocalization() {}

    Relocalization(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        // Initialize the variables and subsribe/advertise topics here
        Initialize();

        // Make thread to monitor the loadm instances
        checkLoamThread = std::thread(&Relocalization::CheckLiteLoams, this);
    }

    void CheckLiteLoams()
    {
        while(ros::ok())
        {
            {
                lock_guard<mutex> lg(loam_mtx);

                // Iterate through existing instances to check their status
                for (size_t lidx = 0; lidx < loamInstances.size(); ++lidx)
                {
                    auto &loam = loamInstances[lidx];

                    if (loam == nullptr)
                        continue;

                    // Restart the instance if it has exceeded 10 seconds and is not working
                    if (loam->timeSinceStart() > 10 && !loam->loamConverged() && loam->isRunning())
                    {
                        ROS_INFO("[Relocalization] LITELOAM %d exceeded 10 sec and is not working. Restarting...",
                                loam->getID());
                                            
                        loam->stop();
                    }
                    else if(loam->loamConverged())
                    {
                        // Do something to begin relocalization 
                    }
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    void Initialize()
    {
        // ULOC pose subcriber
        ulocSub = nh_ptr->subscribe("/uwb_pose", 10, &Relocalization::ULOCCallback, this);

        // loadPriorMap
        string prior_map_dir = "";
        nh_ptr->param("/prior_map_dir", prior_map_dir, string(""));
        ROS_INFO("prior_map_dir: %s", prior_map_dir.c_str());

        this->priorMap.reset(new pcl::PointCloud<pcl::PointXYZI>());
        this->kdTreeMap.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());

        std::string pcd_file = prior_map_dir + "/priormap.pcd";
        ROS_INFO("Prebuilt pcd map found, loading...");

        pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *(this->priorMap));
        ROS_INFO("Prior Map (%zu points).", priorMap->size());

        double priormap_viz_res = 0.2;
        pcl::UniformSampling<pcl::PointXYZI> downsampler;
        downsampler.setInputCloud(this->priorMap);
        downsampler.setRadiusSearch(priormap_viz_res);

        // Create a new point cloud to store the downsampled map
        pcl::PointCloud<pcl::PointXYZI>::Ptr downsampledMap(new pcl::PointCloud<pcl::PointXYZI>());
        downsampler.filter(*downsampledMap);

        // Assign the downsampled map to priorMap
        this->priorMap = downsampledMap;

        ROS_INFO("Downsampled Prior Map (%zu points).", priorMap->size());

        // Update kdTree with the new downsampled map
        this->kdTreeMap->setInputCloud(this->priorMap);
        priorMapReady = true;

        ROS_INFO("Prior Map Load Completed \n");
    }

    void ULOCCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        if (!priorMapReady)
        {
            ROS_WARN("[Relocalization] Prior map is not ready, skipping ULOCCallback.");
            return;
        }

        mytf pose(*msg);

        {
            lock_guard<mutex> lg(loam_mtx);

            // If the number of LITELOAM instances is less than 10, create one more
            if (loamInstances.size() < 10)
            {
                int newID = loamInstances.size(); // Assign a new ID based on the current size
                auto newLoam = std::make_shared<LITELOAM>(priorMap, kdTreeMap, pose, newID, nh_ptr);
                loamInstances.push_back(newLoam);
                ROS_INFO("[Relocalization] Created new LITELOAM instance with ID %d. "
                        "Total instances: %lu",
                        newID, loamInstances.size());
            }

            for (size_t lidx = 0; lidx < loamInstances.size(); ++lidx)
            {
                auto &loam = loamInstances[lidx];

                // Add a new loam if the current loam is null
                if (!loam->isRunning())
                {
                    // Replace the instance with a new one using the same ID
                    loam = std::make_shared<LITELOAM>(priorMap, kdTreeMap, pose, loam->getID(), nh_ptr);
                    ROS_INFO("[Relocalization] LITELOAM %d restarted.", loam->getID());

                    break;
                }
            }
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "relocalization");
    ros::NodeHandle nh("~");
    ros::NodeHandlePtr nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    ROS_INFO(KGRN "----> Relocalization Started." RESET);

    Relocalization relocalization(nh_ptr);

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    return 0;
}