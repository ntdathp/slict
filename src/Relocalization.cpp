#include "STDesc.h"
#include "slict/FeatureCloud.h"
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
#include <factor/PreintFactor.h>

// For brevity
typedef sensor_msgs::PointCloud2::ConstPtr rosCloudMsgPtr;
typedef sensor_msgs::PointCloud2 rosCloudMsg;
typedef Sophus::SO3d SO3d;

class LITELOAM {
private:
  ros::NodeHandlePtr nh_ptr;

  // Subscribers
  ros::Subscriber lidarCloudSub;
  ros::Subscriber imuSub;

  // Publishers
  ros::Publisher relocPub;
  ros::Publisher alignedCloudPub;

  // Queues + mutex
  std::deque<CloudXYZIPtr> cloud_queue;
  std::deque<sensor_msgs::ImuConstPtr> imu_queue;
  std::mutex cloud_mutex;
  std::mutex imu_mutex;

  // Prior map + kdtree
  CloudXYZIPtr priorMap;
  KdFLANNPtr kdTreeMap;

  // Main thread
  std::thread processThread;
  bool running = true;

  // Start time
  double startTime;

  // Initial pose
  mytf initPose;
  int liteloam_id;

  // Keep track of last cloud time
  double last_cloud_time_ = -1.0;

  // Pose from previous iteration => Pose_i
  Eigen::Vector3d prev_t;
  Eigen::Quaterniond prev_q;

public:
  // Destructor
  ~LITELOAM() {
    running = false;
    if (processThread.joinable())
      processThread.join();
    ROS_WARN("liteloam %d destructed.", liteloam_id);
  }

  // Constructor
  LITELOAM(const CloudXYZIPtr &pm, const KdFLANNPtr &kdt, const mytf &initPose_,
           int id, const ros::NodeHandlePtr &nh)
      : priorMap(pm), kdTreeMap(kdt), initPose(initPose_), liteloam_id(id),
        nh_ptr(nh) {
    // Subscribe to /lastcloud and /vn100/imu
    lidarCloudSub =
        nh_ptr->subscribe("/lastcloud", 100, &LITELOAM::PCHandler, this);
    imuSub = nh_ptr->subscribe("/vn100/imu", 500, &LITELOAM::IMUHandler, this);

    // Publishers
    relocPub =
        nh_ptr->advertise<geometry_msgs::PoseStamped>("/liteloam_pose", 100);
    alignedCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(
        "/liteloam_aligned_cloud", 1);

    startTime = ros::Time::now().toSec();
    running = true;
    processThread = std::thread(&LITELOAM::processBuffer, this);

    // Initialize prev pose from initPose
    prev_t = initPose.pos;
    prev_q = initPose.rot;

    ROS_INFO_STREAM("[LITELOAM " << liteloam_id
                                 << "] Constructed - thread started");
  }

  // Stop
  void stop() { running = false; }

  // IMU callback
  void IMUHandler(const sensor_msgs::ImuConstPtr &msg) {
    std::lock_guard<std::mutex> lock(imu_mutex);
    imu_queue.push_back(msg);
  }

  // Lidar callback
  void PCHandler(const sensor_msgs::PointCloud2ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(cloud_mutex);
    CloudXYZIPtr newC(new CloudXYZI());
    pcl::fromROSMsg(*msg, *newC);
    cloud_queue.push_back(newC);
  }

  void processBuffer() {
    while (running && ros::ok()) {
      // 1) Check if any cloud is in the queue
      if (cloud_queue.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      // 2) Retrieve the next cloud from the queue
      CloudXYZIPtr cloudToProcess;
      {
        std::lock_guard<std::mutex> lk(cloud_mutex);
        cloudToProcess = cloud_queue.front();
        cloud_queue.pop_front();
      }

      // Convert timestamp from PCL header format to ros::Time
      uint64_t pcl_stamp = cloudToProcess->header.stamp;
      ros::Time cloudStamp(pcl_stamp / 1000000ULL,
                           (pcl_stamp % 1000000ULL) * 1000ULL);
      double currentCloudTime = cloudStamp.toSec();

      // Keep track of the previous cloud’s time for IMU integration bounds
      double prevCloudTime =
          (last_cloud_time_ < 0.0 ? currentCloudTime : last_cloud_time_);
      last_cloud_time_ = currentCloudTime;

      // For demonstration, print when liteloam_id == 0
      if (liteloam_id == 0) {
        ROS_INFO("liteloam %d run=%.3f", liteloam_id, timeSinceStart());
      }

      // ----------------------------------------------------------------------
      // (A) Downsample the input point cloud
      pcl::PointCloud<PointXYZI>::Ptr src(new pcl::PointCloud<PointXYZI>());
      pcl::copyPointCloud(*cloudToProcess, *src);

      pcl::VoxelGrid<PointXYZI> vg;
      vg.setInputCloud(src);
      vg.setLeafSize(0.3f, 0.3f, 0.3f);
      pcl::PointCloud<PointXYZI>::Ptr srcFiltered(
          new pcl::PointCloud<PointXYZI>());
      vg.filter(*srcFiltered);

      // ----------------------------------------------------------------------
      // (B) Collect IMU messages in the time window [prevCloudTime,
      // currentCloudTime]
      std::deque<sensor_msgs::ImuConstPtr> localIMU;
      {
        std::lock_guard<std::mutex> lk(imu_mutex);
        while (!imu_queue.empty()) {
          double t_imu = imu_queue.front()->header.stamp.toSec();
          if (t_imu >= prevCloudTime && t_imu <= currentCloudTime) {
            localIMU.push_back(imu_queue.front());
            imu_queue.pop_front();
          } else if (t_imu < prevCloudTime) {
            imu_queue.pop_front();
          } else {
            break;
          }
        }
      }

      // ----------------------------------------------------------------------
      // (C) Create an IMU preintegration object if IMU data is available
      PreintBase *preint_imu = nullptr;
      if (!localIMU.empty()) {
        // Example noise parameters (adjust them according to your sensor spec)
        double ACC_N = 0.6, ACC_W = 0.08;
        double GYR_N = 0.05, GYR_W = 0.003;
        Eigen::Vector3d GRAV(0, 0, 9.81);

        // Assume zero bias for demonstration (or retrieve from your system)
        Eigen::Vector3d initBa(0, 0, 0);
        Eigen::Vector3d initBg(0, 0, 0);

        // Initialize the preintegration with the first IMU reading
        Eigen::Vector3d firstAcc(localIMU.front()->linear_acceleration.x,
                                 localIMU.front()->linear_acceleration.y,
                                 localIMU.front()->linear_acceleration.z);
        Eigen::Vector3d firstGyr(localIMU.front()->angular_velocity.x,
                                 localIMU.front()->angular_velocity.y,
                                 localIMU.front()->angular_velocity.z);

        preint_imu =
            new PreintBase(firstAcc, firstGyr, initBa, initBg, true, ACC_N,
                           ACC_W, GYR_N, GYR_W, GRAV, liteloam_id);

        // Integrate the rest of the IMU measurements
        for (size_t k = 1; k < localIMU.size(); k++) {
          double dt = localIMU[k]->header.stamp.toSec() -
                      localIMU[k - 1]->header.stamp.toSec();

          Eigen::Vector3d aK(localIMU[k]->linear_acceleration.x,
                             localIMU[k]->linear_acceleration.y,
                             localIMU[k]->linear_acceleration.z);
          Eigen::Vector3d gK(localIMU[k]->angular_velocity.x,
                             localIMU[k]->angular_velocity.y,
                             localIMU[k]->angular_velocity.z);

          preint_imu->push_back(dt, aK, gK);
        }
      }

      // ----------------------------------------------------------------------
      // (D) Perform ICP using the downsampled cloud srcFiltered
      pcl::IterativeClosestPoint<PointXYZI, PointXYZI> icp;
      icp.setInputSource(srcFiltered);
      icp.setInputTarget(priorMap);
      icp.setMaxCorrespondenceDistance(20.0);
      icp.setMaximumIterations(10);
      icp.setTransformationEpsilon(1e-6);
      icp.setEuclideanFitnessEpsilon(1e-6);

      // Use the previous quaternion to estimate an initial yaw
      double roll, pitch, yaw;
      {
        tf::Matrix3x3 Rq(
            tf::Quaternion(prev_q.x(), prev_q.y(), prev_q.z(), prev_q.w()));
        Rq.getRPY(roll, pitch, yaw);
      }

      double best_fitness = 1e9;
      Eigen::Matrix4f bestTrans = Eigen::Matrix4f::Identity();

      // Scan +/- 20 degrees in yaw (step of 5) to find the best fitness
      for (double d = -20; d <= 20; d += 5) {
        double yaw_f = yaw + d * M_PI / 180.0;
        Eigen::AngleAxisf yawA((float)yaw_f, Eigen::Vector3f::UnitZ());
        Eigen::Matrix3f Rg =
            prev_q.cast<float>().toRotationMatrix() * yawA.toRotationMatrix();
        Eigen::Vector3f Tg = prev_t.cast<float>();

        Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
        guess.block<3, 3>(0, 0) = Rg;
        guess.block<3, 1>(0, 3) = Tg;

        pcl::PointCloud<PointXYZI>::Ptr aligned(
            new pcl::PointCloud<PointXYZI>());
        icp.align(*aligned, guess);

        if (icp.hasConverged()) {
          double fit = icp.getFitnessScore();
          if (fit < best_fitness) {
            best_fitness = fit;
            bestTrans = icp.getFinalTransformation();
          }
        }
      }

      // If ICP did not converge, skip this iteration
      if (best_fitness >= 1e9) {
        ROS_WARN("ICP not converge. best_fitness=%.3f", best_fitness);
        if (preint_imu) {
          delete preint_imu;
        }
        continue;
      }

      // Convert bestTrans to a pose (td, qd)
      Eigen::Matrix3f Rb = bestTrans.block<3, 3>(0, 0);
      Eigen::Vector3f tb = bestTrans.block<3, 1>(0, 3);
      Eigen::Quaternionf qb(Rb);
      Eigen::Quaterniond qd(qb.w(), qb.x(), qb.y(), qb.z());
      Eigen::Vector3d td(tb.x(), tb.y(), tb.z());

      // ----------------------------------------------------------------------
      // (E) Now incorporate IMU factor via ceres
      Eigen::Vector3d final_t;
      Eigen::Quaterniond final_q;

      if (preint_imu) {
        // Prepare parameter blocks for pose_i, pose_j
        double param_pose_i[7] = {prev_t.x(), prev_t.y(), prev_t.z(),
                                  prev_q.x(), prev_q.y(), prev_q.z(),
                                  prev_q.w()};
        double param_vel_i[3] = {0, 0, 0};
        double param_bias_i[6] = {0, 0, 0, 0, 0, 0};

        double param_pose_j[7] = {td.x(), td.y(), td.z(), qd.x(),
                                  qd.y(), qd.z(), qd.w()};
        double param_vel_j[3] = {0, 0, 0};
        double param_bias_j[6] = {0, 0, 0, 0, 0, 0};

        // Build the Ceres problem
        ceres::Problem problem;
        problem.AddParameterBlock(param_pose_i, 7,
                                  new PoseLocalParameterization());
        problem.AddParameterBlock(param_vel_i, 3);
        problem.AddParameterBlock(param_bias_i, 6);

        problem.AddParameterBlock(param_pose_j, 7,
                                  new PoseLocalParameterization());
        problem.AddParameterBlock(param_vel_j, 3);
        problem.AddParameterBlock(param_bias_j, 6);

        // Create the IMU factor and add it to the problem
        PreintFactor *imu_factor = new PreintFactor(preint_imu);
        problem.AddResidualBlock(imu_factor, nullptr, param_pose_i, param_vel_i,
                                 param_bias_i, param_pose_j, param_vel_j,
                                 param_bias_j);

        // Solve
        ceres::Solver::Options opt;
        opt.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summ;
        ceres::Solve(opt, &problem, &summ);
        ROS_INFO_STREAM("IMU factor => " << summ.BriefReport());

        // Retrieve optimized results
        final_t << param_pose_j[0], param_pose_j[1], param_pose_j[2];
        final_q = Eigen::Quaterniond(param_pose_j[6], param_pose_j[3],
                                     param_pose_j[4], param_pose_j[5])
                      .normalized();

        delete imu_factor;
      } else {
        // If no IMU data is available, just use the ICP result
        final_t = td;
        final_q = qd;
      }

      // Update the previous pose for the next iteration
      prev_t = final_t;
      prev_q = final_q;

      if (preint_imu) {
        delete preint_imu;
        preint_imu = nullptr;
      }

      // ----------------------------------------------------------------------
      // **Publish** (A) pose and (B) aligned cloud **after** IMU optimization
      {
        // (A) Publish pose
        geometry_msgs::PoseStamped ps;
        ps.header.stamp = cloudStamp;
        ps.header.frame_id = "map";
        ps.pose.position.x = final_t.x();
        ps.pose.position.y = final_t.y();
        ps.pose.position.z = final_t.z();
        ps.pose.orientation.x = final_q.x();
        ps.pose.orientation.y = final_q.y();
        ps.pose.orientation.z = final_q.z();
        ps.pose.orientation.w = final_q.w();
        relocPub.publish(ps);

        // (B) Publish the aligned cloud using the final transformation
        Eigen::Matrix4f finalTrans = Eigen::Matrix4f::Identity();
        finalTrans.block<3, 3>(0, 0) = final_q.cast<float>().toRotationMatrix();
        finalTrans.block<3, 1>(0, 3) = final_t.cast<float>();

        pcl::PointCloud<PointXYZI>::Ptr alignedC(
            new pcl::PointCloud<PointXYZI>());
        pcl::transformPointCloud(*srcFiltered, *alignedC, finalTrans);

        sensor_msgs::PointCloud2 pc2;
        pcl::toROSMsg(*alignedC, pc2);
        pc2.header.stamp = cloudStamp;
        pc2.header.frame_id = "map";
        alignedCloudPub.publish(pc2);
      }

    } // end while(running && ros::ok())

    ROS_INFO("liteloam %d exit", liteloam_id);
  }

  void Associate(const KdFLANNPtr &kdtreeMap, const CloudXYZIPtr &priormap,
                 const CloudXYZITPtr &cloudRaw, const CloudXYZIPtr &cloudInB,
                 const CloudXYZIPtr &cloudInW, vector<LidarCoef> &Coef) {
    ROS_ASSERT_MSG(cloudRaw->size() == cloudInB->size(),
                   "cloudRaw: %d. cloudInB: %d", cloudRaw->size(),
                   cloudInB->size());

    int knnSize = 6;
    double minKnnSqDis = 0.5 * 0.5;
    double min_planarity = 0.2, max_plane_dis = 0.3;

    if (priormap->size() > knnSize) {
      int pointsCount = cloudInW->points.size();
      vector<LidarCoef> Coef_;
      Coef_.resize(pointsCount);

#pragma omp parallel for num_threads(MAX_THREADS)
      for (int pidx = 0; pidx < pointsCount; pidx++) {
        double tpoint = cloudRaw->points[pidx].t;
        PointXYZIT pointRaw = cloudRaw->points[pidx];
        PointXYZI pointInB = cloudInB->points[pidx];
        PointXYZI pointInW = cloudInW->points[pidx];

        Coef_[pidx].n = Vector4d(0, 0, 0, 0);
        Coef_[pidx].t = -1;

        if (!Util::PointIsValid(pointInB)) {
          pointInB.x = 0;
          pointInB.y = 0;
          pointInB.z = 0;
          pointInB.intensity = 0;
          continue;
        }

        if (!Util::PointIsValid(pointInW))
          continue;

        // if (!traj->TimeInInterval(tpoint, 1e-6))
        //     continue;

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
                           Coef_[pidx].n, Coef_[pidx].plnrty)) {
          // ROS_ASSERT(tpoint >= 0);
          Coef_[pidx].t = tpoint;
          Coef_[pidx].f = Vector3d(pointRaw.x, pointRaw.y, pointRaw.z);
          Coef_[pidx].finW = Vector3d(pointInW.x, pointInW.y, pointInW.z);
          Coef_[pidx].fdsk = Vector3d(pointInB.x, pointInB.y, pointInB.z);
        }
      }

      // Copy the coefficients to the buffer
      Coef.clear();
      int totalFeature = 0;
      for (int pidx = 0; pidx < pointsCount; pidx++) {
        LidarCoef &coef = Coef_[pidx];
        if (coef.t >= 0) {
          Coef.push_back(coef);
          Coef.back().ptIdx = totalFeature;
          totalFeature++;
        }
      }
    }
  }

  double timeSinceStart() const { return ros::Time::now().toSec() - startTime; }

  bool loamConverged() const { return false; }
  bool isRunning() const { return running; }
  int getID() const { return liteloam_id; }
};

//====================================================================
// Class Relocalization
//====================================================================
class Relocalization {
private:
  ros::NodeHandlePtr nh_ptr;

  // Subscribers
  ros::Subscriber lidarCloudSub;
  ros::Subscriber ulocSub;

  // Publisher
  ros::Publisher relocPub;

  CloudXYZIPtr priorMap;
  KdFLANNPtr kdTreeMap;
  bool priorMapReady = false;

  std::mutex loam_mtx;
  std::vector<std::shared_ptr<LITELOAM>> loamInstances;
  std::thread checkLoamThread;

public:
  ~Relocalization() {}

  Relocalization(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_) {
    Initialize();
    checkLoamThread = std::thread(&Relocalization::CheckLiteLoams, this);
  }

  void CheckLiteLoams() {
    while (ros::ok()) {
      {
        std::lock_guard<std::mutex> lg(loam_mtx);

        // Iterate through all LITELOAM instances
        for (size_t lidx = 0; lidx < loamInstances.size(); ++lidx) {
          auto &loam = loamInstances[lidx];
          if (!loam)
            continue;

          // If one instance has been running too long but not converged,
          // restart
          if (loam->timeSinceStart() > 10 && !loam->loamConverged() &&
              loam->isRunning()) {
            ROS_INFO("[Relocalization] LITELOAM %d exceeded 10s. Restart...",
                     loam->getID());
            loam->stop();
          } else if (loam->loamConverged()) {
            // do something else
          }
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  void Initialize() {
    // ULOC pose subcriber
    ulocSub =
        nh_ptr->subscribe("/uwb_pose", 10, &Relocalization::ULOCCallback, this);

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
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampledMap(
        new pcl::PointCloud<pcl::PointXYZI>());
    downsampler.filter(*downsampledMap);

    // Assign the downsampled map to priorMap
    this->priorMap = downsampledMap;

    ROS_INFO("Downsampled Prior Map (%zu points).", priorMap->size());

    // Update kdTree with the new downsampled map
    this->kdTreeMap->setInputCloud(this->priorMap);
    priorMapReady = true;

    ROS_INFO("Prior Map Load Completed \n");
  }

  void ULOCCallback(const geometry_msgs::PoseStamped::ConstPtr &msg) {
    if (!priorMapReady) {
      ROS_WARN("[Relocalization] Prior map is not ready.");
      return;
    }

    mytf pose(*msg);
    {
      std::lock_guard<std::mutex> lg(loam_mtx);

      // If we have fewer than 10 LITELOAM instances, create a new one
      if (loamInstances.size() < 10) {
        int newID = loamInstances.size();
        auto newLoam = std::make_shared<LITELOAM>(priorMap, kdTreeMap, pose,
                                                  newID, nh_ptr);
        loamInstances.push_back(newLoam);

        ROS_INFO("[Relocalization] Created LITELOAM ID=%d. total=%lu", newID,
                 loamInstances.size());
      }

      // If an existing instance is not running, restart it with the same ID
      for (size_t lidx = 0; lidx < loamInstances.size(); ++lidx) {
        auto &loam = loamInstances[lidx];
        if (!loam->isRunning()) {
          loam = std::make_shared<LITELOAM>(priorMap, kdTreeMap, pose,
                                            loam->getID(), nh_ptr);
          ROS_INFO("[Relocalization] LITELOAM %d restarted.", loam->getID());
          break;
        }
      }
    }
  }
};

//====================================================================
// main
//====================================================================
int main(int argc, char **argv) {
  ros::init(argc, argv, "relocalization");
  ros::NodeHandle nh("~");
  ros::NodeHandlePtr nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

  ROS_INFO(KGRN "----> Relocalization Started." RESET);

  Relocalization relocalization(nh_ptr);

  ros::MultiThreadedSpinner spinner(0);
  spinner.spin();
  return 0;
}
