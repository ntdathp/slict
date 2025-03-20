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
    ros::Publisher alignedCloudPub;

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

        alignedCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/liteloam_aligned_cloud", 1);

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
            ros::Time cloudTimestamp = ros::Time(
                pcl_timestamp / 1e6,
                (pcl_timestamp % static_cast<uint64_t>(1e6)) * 1e3);
    
            if (liteloam_id == 0)
            {
                ROS_INFO("liteloam_id %d. Start time: %f. Running Time: %f.",
                         liteloam_id, startTime, timeSinceStart());
            }
    
            //------------------------------------------------
            // 1) Pre-processing of input cloud
            //------------------------------------------------
            pcl::PointCloud<pcl::PointXYZI>::Ptr sourceCloud(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::copyPointCloud(*cloudToProcess, *sourceCloud);
    
            // Apply VoxelGrid filter to downsample the point cloud
            pcl::VoxelGrid<pcl::PointXYZI> vg;
            vg.setInputCloud(sourceCloud);
            vg.setLeafSize(0.3f, 0.3f, 0.3f);
            pcl::PointCloud<pcl::PointXYZI>::Ptr sourceFiltered(new pcl::PointCloud<pcl::PointXYZI>());
            vg.filter(*sourceFiltered);
    
            //------------------------------------------------
            // 2) Compute yaw from initPose
            //------------------------------------------------
            double roll, pitch, yaw_init;
            {
                tf::Matrix3x3(
                    tf::Quaternion(initPose.rot.x(),
                                   initPose.rot.y(),
                                   initPose.rot.z(),
                                   initPose.rot.w()))
                    .getRPY(roll, pitch, yaw_init);
            }
    
            // Between init [-20°, 20°], step 5°
            std::vector<double> yaw_candidates;
            for (double delta = -20.0; delta <= 20.0; delta += 5.0)
            {
                double yaw_rad = (yaw_init + (delta * M_PI / 180.0));
                yaw_candidates.push_back(yaw_rad);
            }
    
            //------------------------------------------------
            // 3) Iterate through yaw candidates & run ICP
            //------------------------------------------------
            double best_fitness = std::numeric_limits<double>::max();
            Eigen::Matrix4f best_trans = Eigen::Matrix4f::Identity();
    
            // ICP
            pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
            icp.setInputSource(sourceFiltered); // moving cloud
            icp.setInputTarget(priorMap);       // static map
    
            // ICP parameters 
            icp.setMaxCorrespondenceDistance(20.0);
            icp.setMaximumIterations(10);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
    
            //initPose (Eigen::Quaternion -> Matrix3f)
            Eigen::Matrix3f R_init = initPose.rot.toRotationMatrix().cast<float>();
            // 
            Eigen::Vector3f t_init = initPose.pos.cast<float>();
    
            // Multiple yaw
            for (double yaw_candidate : yaw_candidates)
            {
                
                float yaw_f = static_cast<float>(yaw_candidate);
    
                Eigen::AngleAxisf yawAngle(yaw_f, Eigen::Vector3f::UnitZ());
    
                Eigen::Matrix3f R_candidate = R_init * yawAngle.toRotationMatrix();
    
                // Initial guess
                Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
                guess.block<3,3>(0,0) = R_candidate;
                guess.block<3,1>(0,3) = t_init;
    
                // Run ICP
                pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());
                icp.align(*aligned, guess);
    
                double fitness = icp.getFitnessScore();
                if (icp.hasConverged() && fitness < best_fitness)
                {
                    best_fitness = fitness;
                    best_trans = icp.getFinalTransformation();
                }
            }
    
            //------------------------------------------------
            // 4) Publish the best ICP result
            //------------------------------------------------
            double icpFitnessThres = 0.3; 
            if (best_fitness < std::numeric_limits<double>::max()) //&& best_fitness < icpFitnessThres 
            {
                ROS_INFO_STREAM("ICP converged. Best fitness: " << best_fitness);
    
                Eigen::Matrix3f R_best = best_trans.block<3,3>(0,0);
                Eigen::Vector3f t_best = best_trans.block<3,1>(0,3);
                Eigen::Quaternionf q_best(R_best);
    
                geometry_msgs::PoseStamped pose_msg;

                ros::Duration processingDuration = ros::Time::now() - startProcessingTime;
                pose_msg.header.stamp = cloudTimestamp + processingDuration;
                pose_msg.header.frame_id = "map";
    
                pose_msg.pose.position.x = t_best.x();
                pose_msg.pose.position.y = t_best.y();
                pose_msg.pose.position.z = t_best.z();
    
                pose_msg.pose.orientation.x = q_best.x();
                pose_msg.pose.orientation.y = q_best.y();
                pose_msg.pose.orientation.z = q_best.z();
                pose_msg.pose.orientation.w = q_best.w();
    
                relocPub.publish(pose_msg);

                pcl::PointCloud<pcl::PointXYZI>::Ptr alignedCloud(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::transformPointCloud(*sourceFiltered, *alignedCloud, best_trans);
            
                sensor_msgs::PointCloud2 alignedMsg;
                pcl::toROSMsg(*alignedCloud, alignedMsg);
            
                // Set header cho Rviz
                alignedMsg.header.stamp = pose_msg.header.stamp; // hoặc cloudTimestamp
                alignedMsg.header.frame_id = "map";
            
                alignedCloudPub.publish(alignedMsg);
            }
            else
            {
                ROS_WARN("ICP did not converge with any yaw candidate. Best fitness: %f", best_fitness);
            }
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

        double priormap_viz_res = 1;
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