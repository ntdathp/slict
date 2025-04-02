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

/* All needed for filter of custom point type */
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
#include "PreintBase.h"

// Shorthands
typedef sensor_msgs::PointCloud2::ConstPtr rosCloudMsgPtr;
typedef sensor_msgs::PointCloud2 rosCloudMsg;
typedef Sophus::SO3d SO3d;

// Shorthands for ufomap
namespace ufopred = ufo::map::predicate;
using ufoSurfelMap    = ufo::map::SurfelMap;
using ufoSurfelMapPtr = boost::shared_ptr<ufoSurfelMap>;
using ufoNode         = ufo::map::NodeBV;
using ufoSphere       = ufo::geometry::Sphere;
using ufoPoint3       = ufo::map::Point3;

class LITELOAM
{
private:
    ros::NodeHandlePtr nh_ptr;

    // Subscribers
    ros::Subscriber lidarCloudSub;
    ros::Subscriber imuSub;

    // Publishers
    ros::Publisher relocPub;
    ros::Publisher alignedCloudPub;

    // Queues + mutex
    std::deque<CloudXYZIPtr>          cloud_queue;  // cloud queue
    std::deque<sensor_msgs::ImuConstPtr> imu_queue; // IMU queue

    std::mutex cloud_mutex;   // protects cloud_queue
    std::mutex imu_mutex;     // protects imu_queue

    // Clouds on the sliding window
    deque<CloudXYZIPtr> SwCloud;

    // Prior map + kd-tree
    CloudXYZIPtr priorMap;
    KdFLANNPtr   kdTreeMap;

    // Thread for main processing
    std::thread processThread;
    bool running = true;

    // Start time
    double startTime;

    // Initial pose
    mytf initPose;
    
    // ID of this LITELOAM instance
    int liteloam_id;

    // Remember the previous cloud's timestamp
    double last_cloud_time_ = -1.0;

public:

    ~LITELOAM()
    {
        // Stop the main loop
        running = false;
        if (processThread.joinable())
            processThread.join();

        ROS_WARN("liteloam %d destructed." RESET, liteloam_id);
    }

    LITELOAM(const CloudXYZIPtr &priorMap_,
             const KdFLANNPtr &kdTreeMap_,
             const mytf &initPose_,
             int id,
             const ros::NodeHandlePtr &nh_ptr_)
        : priorMap(priorMap_),
          kdTreeMap(kdTreeMap_),
          initPose(initPose_),
          liteloam_id(id),
          nh_ptr(nh_ptr_)
    {
        // Subscribe to LiDAR
        lidarCloudSub = nh_ptr->subscribe(
            "/lastcloud", 100, &LITELOAM::PCHandler, this);

        // Subscribe to IMU
        imuSub = nh_ptr->subscribe(
            "/vn100/imu", 500, &LITELOAM::IMUHandler, this);

        // Advertised topics
        relocPub = nh_ptr->advertise<geometry_msgs::PoseStamped>(
            "/liteloam_pose", 100);

        alignedCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(
            "/liteloam_aligned_cloud", 1);

        std::cout << "[LITELOAM] " << liteloam_id
                  << " Subscribed to /lastcloud + /vn100/imu."
                  << " Publishing to /liteloam_pose"
                  << std::endl;

        startTime = ros::Time::now().toSec();
        running   = true;

        // Start the main thread
        processThread = std::thread(&LITELOAM::processBuffer, this);
        ROS_INFO_STREAM("[LITELOAM " << liteloam_id << "] Constructor - thread started");
    }

    void stop() { running = false; }

    //--------------------------------------------------------------------------
    // (1) IMU callback
    //--------------------------------------------------------------------------
    void IMUHandler(const sensor_msgs::ImuConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(imu_mutex);
        imu_queue.push_back(msg);
    }

    //--------------------------------------------------------------------------
    // (2) LiDAR callback
    //--------------------------------------------------------------------------
    void PCHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(cloud_mutex);

        CloudXYZIPtr newCloud(new CloudXYZI());
        pcl::fromROSMsg(*msg, *newCloud);

        cloud_queue.push_back(newCloud);
    }

    //--------------------------------------------------------------------------
    // (3) The main processing thread
    //--------------------------------------------------------------------------
    void processBuffer()
    {
        while (running && ros::ok())
        {
            // Wait for new clouds
            if (cloud_queue.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
    
            // Get the newest cloud from the queue
            CloudXYZIPtr cloudToProcess;
            {
                std::lock_guard<std::mutex> lock(cloud_mutex);
                cloudToProcess = cloud_queue.front();
                cloud_queue.pop_front();
            }

            // Determine the timestamp from header.stamp
            uint64_t pcl_timestamp = cloudToProcess->header.stamp;
            ros::Time cloud_ros_time(
                pcl_timestamp / (uint64_t)1e6,
                (pcl_timestamp % (uint64_t)1e6) * (uint64_t)1e3);

            double currentCloudTime = cloud_ros_time.toSec();
            // If it's the first cloud, last_cloud_time_ might be -1
            double prevCloudTime = (last_cloud_time_ < 0.0)
                                   ? currentCloudTime
                                   : last_cloud_time_;
            // Update last_cloud_time_ for the next iteration
            last_cloud_time_ = currentCloudTime;

            if (liteloam_id == 0)
            {
                ROS_INFO("liteloam_id %d. Start time: %.3f. Running Time: %.3f.",
                         liteloam_id, startTime, timeSinceStart());
            }

            //-----------------------------------------------------------------
            // [1] Downsample the incoming cloud
            //-----------------------------------------------------------------
            pcl::PointCloud<PointXYZI>::Ptr src(new pcl::PointCloud<PointXYZI>());
            pcl::copyPointCloud(*cloudToProcess, *src);

            pcl::VoxelGrid<PointXYZI> vg;
            vg.setInputCloud(src);
            vg.setLeafSize(0.3f, 0.3f, 0.3f);
            pcl::PointCloud<PointXYZI>::Ptr srcFiltered(new pcl::PointCloud<PointXYZI>());
            vg.filter(*srcFiltered);

            //-----------------------------------------------------------------
            // [2] Collect the IMU messages within [prevCloudTime, currentCloudTime]
            //-----------------------------------------------------------------
            std::deque<sensor_msgs::ImuConstPtr> localImuBuf;
            {
                std::lock_guard<std::mutex> lock(imu_mutex);

                while (!imu_queue.empty())
                {
                    double t_imu = imu_queue.front()->header.stamp.toSec();
                    if (t_imu >= prevCloudTime && t_imu <= currentCloudTime)
                    {
                        localImuBuf.push_back(imu_queue.front());
                        imu_queue.pop_front();
                    }
                    else if (t_imu < prevCloudTime)
                    {
                        // Too old => discard
                        imu_queue.pop_front();
                    }
                    else
                    {
                        // t_imu > currentCloudTime => break
                        break;
                    }
                }
            }

            //-----------------------------------------------------------------
            // [3] IMU Preintegration (PreintBase)
            //-----------------------------------------------------------------
            double ACC_N = 0.6, ACC_W = 0.08;
            double GYR_N = 0.05, GYR_W = 0.003;
            Eigen::Vector3d GRAV(0,0,9.81);

            // Temporary bias
            Eigen::Vector3d initBa(0,0,0);
            Eigen::Vector3d initBg(0,0,0);

            PreintBase *preint_imu = nullptr;
            if (!localImuBuf.empty())
            {
                // Create the first sample
                Eigen::Vector3d firstAcc(
                    localImuBuf.front()->linear_acceleration.x,
                    localImuBuf.front()->linear_acceleration.y,
                    localImuBuf.front()->linear_acceleration.z
                );
                Eigen::Vector3d firstGyr(
                    localImuBuf.front()->angular_velocity.x,
                    localImuBuf.front()->angular_velocity.y,
                    localImuBuf.front()->angular_velocity.z
                );

                // Build the PreintBase object
                preint_imu = new PreintBase(
                    firstAcc, 
                    firstGyr,
                    initBa, 
                    initBg,
                    true,  // show_init_cost
                    ACC_N, 
                    ACC_W,
                    GYR_N, 
                    GYR_W,
                    GRAV,
                    liteloam_id
                );

                // Push each IMU sample to preint_imu
                for (size_t k=1; k<localImuBuf.size(); k++)
                {
                    double dt = localImuBuf[k]->header.stamp.toSec() 
                              - localImuBuf[k-1]->header.stamp.toSec();

                    Eigen::Vector3d ak(
                        localImuBuf[k]->linear_acceleration.x,
                        localImuBuf[k]->linear_acceleration.y,
                        localImuBuf[k]->linear_acceleration.z
                    );
                    Eigen::Vector3d gk(
                        localImuBuf[k]->angular_velocity.x,
                        localImuBuf[k]->angular_velocity.y,
                        localImuBuf[k]->angular_velocity.z
                    );
                    preint_imu->push_back(dt, ak, gk);
                }
            }


            //-----------------------------------------------------------------
            // [5] ICP alignment (cloud is already deskewed)
            //-----------------------------------------------------------------
            double best_fitness = std::numeric_limits<double>::max();
            Eigen::Matrix4f bestTrans = Eigen::Matrix4f::Identity();

            pcl::IterativeClosestPoint<PointXYZI, PointXYZI> icp;
            icp.setInputSource(srcFiltered);
            icp.setInputTarget(priorMap);
            icp.setMaxCorrespondenceDistance(20.0);
            icp.setMaximumIterations(10);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);

            double roll, pitch, yaw_init;
            {
                tf::Matrix3x3(tf::Quaternion(initPose.rot.x(),
                                             initPose.rot.y(),
                                             initPose.rot.z(),
                                             initPose.rot.w())
                ).getRPY(roll, pitch, yaw_init);
            }

            // Sweep yaw from -20 to +20 deg in steps of 1 deg
            std::vector<double> yaw_candidates;
            for (double d=-20.0; d<=20.0; d+=1.0)
            {
                double yaw_rad = yaw_init + (d*M_PI/180.0);
                yaw_candidates.push_back(yaw_rad);
            }

            // Pose guess from initPose
            Eigen::Matrix3f R_init = initPose.rot.toRotationMatrix().cast<float>();
            Eigen::Vector3f t_init = initPose.pos.cast<float>();

            // Evaluate all yaw candidates
            for (double y_c : yaw_candidates)
            {
                Eigen::AngleAxisf yawAngle((float)y_c, Eigen::Vector3f::UnitZ());
                Eigen::Matrix3f R_cand = R_init * yawAngle.toRotationMatrix();

                Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
                guess.block<3,3>(0,0) = R_cand;
                guess.block<3,1>(0,3) = t_init;

                pcl::PointCloud<PointXYZI>::Ptr aligned(new pcl::PointCloud<PointXYZI>());
                icp.align(*aligned, guess);

                double fitness = icp.getFitnessScore();
                if(icp.hasConverged() && fitness<best_fitness)
                {
                    best_fitness = fitness;
                    bestTrans = icp.getFinalTransformation();
                }
            }

            //-----------------------------------------------------------------
            // [6] Publish results
            //-----------------------------------------------------------------
            if (best_fitness < std::numeric_limits<double>::max())
            {
                ROS_INFO("[LITELOAM %d] ICP converged. fitness=%.3f",
                         liteloam_id, best_fitness);

                Eigen::Matrix3f Rb = bestTrans.block<3,3>(0,0);
                Eigen::Vector3f tb = bestTrans.block<3,1>(0,3);
                Eigen::Quaternionf qb(Rb);

                geometry_msgs::PoseStamped pose_msg;
                pose_msg.header.stamp = cloud_ros_time;
                pose_msg.header.frame_id = "map";

                pose_msg.pose.position.x = tb.x();
                pose_msg.pose.position.y = tb.y();
                pose_msg.pose.position.z = tb.z();

                pose_msg.pose.orientation.x = qb.x();
                pose_msg.pose.orientation.y = qb.y();
                pose_msg.pose.orientation.z = qb.z();
                pose_msg.pose.orientation.w = qb.w();

                relocPub.publish(pose_msg);

                // Transform the cloud to the best pose
                pcl::PointCloud<PointXYZI>::Ptr alignedCloud(new pcl::PointCloud<PointXYZI>());
                pcl::transformPointCloud(*srcFiltered, *alignedCloud, bestTrans);

                sensor_msgs::PointCloud2 alignedMsg;
                pcl::toROSMsg(*alignedCloud, alignedMsg);
                alignedMsg.header.stamp    = cloud_ros_time;
                alignedMsg.header.frame_id = "map";
                alignedCloudPub.publish(alignedMsg);
            }
            else
            {
                ROS_WARN("[LITELOAM %d] ICP did NOT converge. best_fitness=%.3f",
                         liteloam_id, best_fitness);
            }

            // Release preint_imu if we do not need it anymore
            // if (preint_imu)
            //     delete preint_imu;
        }

        ROS_INFO(KRED "liteloam_id %d exits." RESET, liteloam_id);
    }

    //----------------------------------------------------------------------
    // [4] Utility
    //----------------------------------------------------------------------
    double timeSinceStart() const
    {
        return ros::Time::now().toSec() - startTime;
    }

    bool loamConverged() const { return false; }
    bool isRunning()     const { return running; }
    int  getID()         const { return liteloam_id; }

    // Example of an "Associate" function (originally in your code)
    void Associate(const KdFLANNPtr &kdtreeMap,
                   const CloudXYZIPtr &priormap,
                   const CloudXYZITPtr &cloudRaw,
                   const CloudXYZIPtr &cloudInB,
                   const CloudXYZIPtr &cloudInW,
                   std::vector<LidarCoef> &Coef)
    {
        ROS_ASSERT_MSG(cloudRaw->size() == cloudInB->size(),
                       "cloudRaw: %d. cloudInB: %d",
                       cloudRaw->size(), cloudInB->size());
        
        int knnSize = 6;
        double minKnnSqDis = 0.5*0.5;
        double min_planarity = 0.2, max_plane_dis = 0.3;

        if (priormap->size() > knnSize)
        {
            int pointsCount = cloudInW->points.size();
            std::vector<LidarCoef> Coef_;
            Coef_.resize(pointsCount);

            #pragma omp parallel for num_threads(MAX_THREADS)
            for (int pidx = 0; pidx < pointsCount; pidx++)
            {
                double tpoint = cloudRaw->points[pidx].t;
                PointXYZIT pointRaw = cloudRaw->points[pidx];
                PointXYZI pointInB  = cloudInB->points[pidx];
                PointXYZI pointInW  = cloudInW->points[pidx];

                Coef_[pidx].n = Eigen::Vector4d(0, 0, 0, 0);
                Coef_[pidx].t = -1;

                if (!Util::PointIsValid(pointInB))
                    continue;
                if (!Util::PointIsValid(pointInW))
                    continue;

                std::vector<int> knn_idx(knnSize, 0);
                std::vector<float> knn_sq_dis(knnSize, 0);
                kdtreeMap->nearestKSearch(pointInW, knnSize, knn_idx, knn_sq_dis);

                std::vector<PointXYZI> nbrPoints;
                if (knn_sq_dis.back() < minKnnSqDis)
                {
                    for (auto &idx : knn_idx)
                        nbrPoints.push_back(priormap->points[idx]);
                }
                else
                    continue;

                // Fit plane
                if (Util::fitPlane(nbrPoints, min_planarity, max_plane_dis,
                                   Coef_[pidx].n, Coef_[pidx].plnrty))
                {
                    Coef_[pidx].t    = tpoint;
                    Coef_[pidx].f    = Eigen::Vector3d(pointRaw.x, pointRaw.y, pointRaw.z);
                    Coef_[pidx].finW = Eigen::Vector3d(pointInW.x, pointInW.y, pointInW.z);
                    Coef_[pidx].fdsk = Eigen::Vector3d(pointInB.x, pointInB.y, pointInB.z);
                }
            }

            // Copy valid Coef to output
            Coef.clear();
            int totalFeature = 0;
            for (int pidx = 0; pidx < pointsCount; pidx++)
            {
                if (Coef_[pidx].t >= 0)
                {
                    Coef.push_back(Coef_[pidx]);
                    Coef.back().ptIdx = totalFeature++;
                }
            }
        }
    }
};

//====================================================================
// Class Relocalization
//====================================================================
class Relocalization
{
private:
    ros::NodeHandlePtr nh_ptr;

    // Subscribers
    ros::Subscriber lidarCloudSub;
    ros::Subscriber ulocSub;

    // Publisher
    ros::Publisher relocPub;

    CloudXYZIPtr priorMap;
    KdFLANNPtr   kdTreeMap;
    bool priorMapReady = false;

    std::mutex loam_mtx;
    std::vector<std::shared_ptr<LITELOAM>> loamInstances;
    std::thread checkLoamThread;

public:
    ~Relocalization() {}

    Relocalization(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        Initialize();
        checkLoamThread = std::thread(&Relocalization::CheckLiteLoams, this);
    }

    void CheckLiteLoams()
    {
        while(ros::ok())
        {
            {
                std::lock_guard<std::mutex> lg(loam_mtx);

                // Iterate through all LITELOAM instances
                for (size_t lidx = 0; lidx < loamInstances.size(); ++lidx)
                {
                    auto &loam = loamInstances[lidx];
                    if (!loam) 
                        continue;

                    // If one instance has been running too long but not converged, restart
                    if (loam->timeSinceStart() > 10
                        && !loam->loamConverged()
                        && loam->isRunning())
                    {
                        ROS_INFO("[Relocalization] LITELOAM %d exceeded 10s. Restart...",
                                 loam->getID());
                        loam->stop();
                    }
                    else if(loam->loamConverged())
                    {
                        // do something else
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
            ROS_WARN("[Relocalization] Prior map is not ready.");
            return;
        }

        mytf pose(*msg);
        {
            std::lock_guard<std::mutex> lg(loam_mtx);

            // If we have fewer than 10 LITELOAM instances, create a new one
            if (loamInstances.size() < 10)
            {
                int newID = loamInstances.size();
                auto newLoam = std::make_shared<LITELOAM>(
                    priorMap, kdTreeMap, pose, newID, nh_ptr
                );
                loamInstances.push_back(newLoam);

                ROS_INFO("[Relocalization] Created LITELOAM ID=%d. total=%lu",
                         newID, loamInstances.size());
            }

            // If an existing instance is not running, restart it with the same ID
            for (size_t lidx = 0; lidx < loamInstances.size(); ++lidx)
            {
                auto &loam = loamInstances[lidx];
                if (!loam->isRunning())
                {
                    loam = std::make_shared<LITELOAM>(
                        priorMap, kdTreeMap, pose, loam->getID(), nh_ptr
                    );
                    ROS_INFO("[Relocalization] LITELOAM %d restarted.",
                             loam->getID());
                    break;
                }
            }
        }
    }
};

//====================================================================
// main
//====================================================================
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
