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
#include <pcl/registration/transformation_estimation_2D.h>

#include "GaussianProcess.hpp"
#include "PreintBase.h"
#include <factor/PreintFactor.h>

// For brevity
typedef sensor_msgs::PointCloud2::ConstPtr rosCloudMsgPtr;
typedef sensor_msgs::PointCloud2 rosCloudMsg;
typedef Sophus::SO3d SO3d;

class LITELOAM
{
private:
  ros::NodeHandlePtr nh_ptr;

  // Subscribers & publishers
  ros::Subscriber lidarCloudSub;
  ros::Subscriber imuSub;
  ros::Publisher relocPub;
  ros::Publisher alignedCloudPub;

  // Queues + mutex
  std::deque<CloudXYZIPtr> cloud_queue;
  std::deque<sensor_msgs::ImuConstPtr> imu_queue;
  std::mutex cloud_mutex;
  std::mutex imu_mutex;

  // Global prior map + KD‐tree
  CloudXYZIPtr priorMap;
  KdFLANNPtr kdTreeMap;

  // Thread control
  std::thread processThread;
  bool running = true;

  // Timing
  double startTime = 0.0;
  double last_cloud_time = -1.0;

  // Pose state
  mytf initPose;
  mytf prevPose;
  int liteloam_id;

  // IMU pre‐integration
  std::unique_ptr<PreintBase> imuPreint;

  // ICP params
  float fineLeaf = 0.1f;
  float coarseLeaf = 0.2f;
  double fineMaxCorr = 10.0;
  double coarseMaxCorr = 30.0;
  double fitnessThresh = 4.0;
  double fitnessThreshAfter = 2.0;

  bool converged_ = false;
  bool first_cloud_ = true;

public:
  // Constructor
  LITELOAM(const CloudXYZIPtr &pm,
           const KdFLANNPtr &kdt,
           const mytf &initP,
           int id,
           const ros::NodeHandlePtr &nh)
      : nh_ptr(nh),
        priorMap(pm),
        kdTreeMap(kdt),
        initPose(initP),
        prevPose(initP),
        liteloam_id(id),
        fineLeaf(0.1f),
        coarseLeaf(0.2f),
        fineMaxCorr(10.0),
        coarseMaxCorr(30.0),
        fitnessThresh(4.0),
        fitnessThreshAfter(2.0),
        converged_(false),
        first_cloud_(true)
  {
    ros::NodeHandle pnh("~");  // private handle
    pnh.param("fine_leaf",          fineLeaf,          fineLeaf);
    pnh.param("coarse_leaf",        coarseLeaf,        coarseLeaf);
    pnh.param("fine_max_corr",      fineMaxCorr,       fineMaxCorr);
    pnh.param("coarse_max_corr",    coarseMaxCorr,     coarseMaxCorr);
    pnh.param("fitness_thresh",     fitnessThresh,     fitnessThresh);
    pnh.param("fitness_thresh_after", fitnessThreshAfter, fitnessThreshAfter);


    kdTreeMap->setInputCloud(priorMap);
    lidarCloudSub = nh_ptr->subscribe("/lastcloud", 100, &LITELOAM::PCHandler, this);
    imuSub = nh_ptr->subscribe("/vn100/imu", 500, &LITELOAM::IMUHandler, this);
    relocPub = nh_ptr->advertise<geometry_msgs::PoseStamped>("/liteloam_pose", 100);
    alignedCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/liteloam_aligned_cloud", 1);

    startTime = ros::Time::now().toSec();
    running = true;
    processThread = std::thread(&LITELOAM::processBuffer, this);

    ROS_INFO_STREAM("[LITELOAM " << liteloam_id << "] Constructed");
  }

  ~LITELOAM()
  {
    stop();
    if (processThread.joinable())
    {
      processThread.join();
    }
    ROS_WARN("[LITELOAM %d] destructed.", liteloam_id);
  }

  void stop() { running = false; }

  double timeSinceStart() const
  {
    return ros::Time::now().toSec() - startTime;
  }

  bool loamConverged() const { return converged_; }
  bool isRunning() const { return running; }
  int getID() const { return liteloam_id; }

private:
  void IMUHandler(const sensor_msgs::ImuConstPtr &msg)
  {
    std::lock_guard<std::mutex> lk(imu_mutex);
    imu_queue.push_back(msg);
  }

  void PCHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    std::lock_guard<std::mutex> lk(cloud_mutex);
    CloudXYZIPtr cloud(new CloudXYZI());
    pcl::fromROSMsg(*msg, *cloud);
    cloud_queue.push_back(cloud);
  }

  void processBuffer()
  {
    while (running && ros::ok())
    {

      bool printFirst = first_cloud_;
      // 1) wait for cloud
      if (cloud_queue.empty())
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      // 2) pop cloud
      CloudXYZIPtr raw;
      {
        std::lock_guard<std::mutex> lk(cloud_mutex);
        raw = cloud_queue.front();
        cloud_queue.pop_front();
      }

      // 3) timestamp & dt
      uint64_t stamp = raw->header.stamp;
      ros::Time cloudTime(stamp / 1000000ULL, (stamp % 1000000ULL) * 1000ULL);
      double tNow = cloudTime.toSec();
      double tPrev = (last_cloud_time < 0.0 ? tNow : last_cloud_time);
      last_cloud_time = tNow;
      double dt = tNow - tPrev;

      // 4) downsample (fine)
      CloudXYZIPtr srcFine(new CloudXYZI());
      {
        pcl::VoxelGrid<PointXYZI> vg;
        vg.setInputCloud(raw);
        vg.setLeafSize(fineLeaf, fineLeaf, fineLeaf);
        vg.filter(*srcFine);
      }

      // 5) collect IMU
      std::vector<sensor_msgs::ImuConstPtr> imuBuf;
      {
        std::lock_guard<std::mutex> lk(imu_mutex);
        while (!imu_queue.empty())
        {
          double t = imu_queue.front()->header.stamp.toSec();
          if (t < tPrev)
          {
            imu_queue.pop_front();
            continue;
          }
          if (t <= tNow)
          {
            imuBuf.push_back(imu_queue.front());
            imu_queue.pop_front();
          }
          else
            break;
        }
      }

      // 6) IMU pre‐integration
      if (!imuBuf.empty())
      {
        static Eigen::Vector3d ba(0, 0, 0), bg(0, 0, 0);
        Eigen::Vector3d a0(imuBuf.front()->linear_acceleration.x,
                           imuBuf.front()->linear_acceleration.y,
                           imuBuf.front()->linear_acceleration.z);
        Eigen::Vector3d g0(imuBuf.front()->angular_velocity.x,
                           imuBuf.front()->angular_velocity.y,
                           imuBuf.front()->angular_velocity.z);
        if (!imuPreint)
        {
          imuPreint.reset(new PreintBase(
              a0, g0, ba, bg,
              true, 0.6, 0.08,
              0.05, 0.003,
              Eigen::Vector3d(0, 0, 9.81),
              liteloam_id));
        }
        else
        {
          imuPreint->repropagate(ba, bg);
        }

        for (size_t i = 1; i < imuBuf.size(); ++i)
        {
          double dti = imuBuf[i]->header.stamp.toSec() - imuBuf[i - 1]->header.stamp.toSec();
          Eigen::Vector3d ai(imuBuf[i]->linear_acceleration.x,
                             imuBuf[i]->linear_acceleration.y,
                             imuBuf[i]->linear_acceleration.z);
          Eigen::Vector3d gi(imuBuf[i]->angular_velocity.x,
                             imuBuf[i]->angular_velocity.y,
                             imuBuf[i]->angular_velocity.z);
          imuPreint->push_back(dti, ai, gi);
        }
      }

      // --- 6.5) Predict initial pose using IMU pre‑integration ---
      mytf predPose = prevPose;
      if (imuPreint)
      {
        // Retrieve integrated IMU deltas: translation (dp), velocity (dv), rotation (dq)
        Eigen::Vector3d dp, dv;
        Eigen::Quaterniond dq;
        imuPreint->getPredictedDeltas(
            dp,                       // integrated position change
            dv,                       // integrated velocity change (unused here)
            dq,                       // integrated orientation change
            prevPose.rot,             // initial orientation Qi
            Eigen::Vector3d::Zero(),  // initial velocity Vi (zero if not tracked)
            imuPreint->linearized_ba, // bias for accelerometer
            imuPreint->linearized_bg  // bias for gyroscope
        );

        // Apply the delta to previous pose to get the predicted pose
        predPose.pos = prevPose.pos + (prevPose.rot * dp);
        predPose.rot = (prevPose.rot * dq).normalized();
      }

      // 7) ICP against priorMap
      Eigen::Matrix4f bestTrans = Eigen::Matrix4f::Identity();
      double bestFitness = std::numeric_limits<double>::infinity();

      // Prepare downsampled clouds
      CloudXYZIPtr srcCoarse(new CloudXYZI());
      {
        pcl::VoxelGrid<PointXYZI> vg;
        vg.setInputCloud(srcFine);
        vg.setLeafSize(coarseLeaf, coarseLeaf, coarseLeaf);
        vg.filter(*srcCoarse);
      }

      // Base pose
      double yaw0 = predPose.yaw() * M_PI / 180.0;
      Eigen::Matrix3f R0 = predPose.rot.cast<float>().toRotationMatrix();
      Eigen::Vector3f T0 = predPose.pos.cast<float>();

      std::vector<double> yaw_offsets;
      if (first_cloud_)
      {
        for (double d = 0; d < 360; d += 30.0)
          yaw_offsets.push_back(d);
      }
      else
      {
        for (double d = -10; d < 10; d += 5.0)
          yaw_offsets.push_back(d);
      }

      for (double d : yaw_offsets)
      {
        // build guess with yaw offset d
        double yaw = prevPose.yaw() * M_PI / 180.0 + d * M_PI / 180.0;
        Eigen::AngleAxisf rz((float)yaw, Eigen::Vector3f::UnitZ());
        Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
        guess.block<3, 3>(0, 0) = R0 * rz.toRotationMatrix();
        guess.block<3, 1>(0, 3) = T0;

        // Coarse pass
        pcl::IterativeClosestPoint<PointXYZI, PointXYZI> icp1;
        icp1.setInputSource(srcCoarse);
        icp1.setInputTarget(priorMap);
        icp1.setMaxCorrespondenceDistance(coarseMaxCorr);
        icp1.setMaximumIterations(5);
        pcl::PointCloud<PointXYZI> alignedCoarse;
        icp1.align(alignedCoarse, guess);
        if (!icp1.hasConverged())
          continue;
        Eigen::Matrix4f coarseT = icp1.getFinalTransformation();

        // Fine pass
        pcl::IterativeClosestPoint<PointXYZI, PointXYZI> icp2;
        icp2.setInputSource(srcFine);
        icp2.setInputTarget(priorMap);
        icp2.setMaxCorrespondenceDistance(fineMaxCorr);
        icp2.setMaximumIterations(10);
        pcl::PointCloud<PointXYZI> alignedFine;
        icp2.align(alignedFine, coarseT);
        if (!icp2.hasConverged())
          continue;

        double fit = icp2.getFitnessScore();
        if (fit < bestFitness)
        {
          bestFitness = fit;
          bestTrans = icp2.getFinalTransformation();
        }
      }

      double thresh = first_cloud_ ? fitnessThresh : fitnessThreshAfter;
      if (bestFitness >= thresh)
      {
        ROS_WARN("[LITELOAM %d] ICP fitness=%.3f >= %.1f skip",
                 liteloam_id, bestFitness, thresh);
        converged_ = false;
        continue;
      }

      first_cloud_ = false;

      // 8) Build pose & refine with IMU factor
      Eigen::Matrix4d bestTransD = bestTrans.cast<double>();
      mytf finalPose(bestTransD);

      if (imuPreint)
      {
        double pi[7] = {
            prevPose.pos.x(), prevPose.pos.y(), prevPose.pos.z(),
            prevPose.rot.x(), prevPose.rot.y(),
            prevPose.rot.z(), prevPose.rot.w()};
        double pj[7] = {
            finalPose.pos.x(), finalPose.pos.y(), finalPose.pos.z(),
            finalPose.rot.x(), finalPose.rot.y(),
            finalPose.rot.z(), finalPose.rot.w()};
        double vi[3] = {0, 0, 0}, vj[3] = {0, 0, 0};
        double bai[6] = {0, 0, 0, 0, 0, 0}, baj[6] = {0, 0, 0, 0, 0, 0};

        ceres::Problem problem;
        problem.AddParameterBlock(pi, 7, new PoseLocalParameterization());
        problem.AddParameterBlock(vi, 3);
        problem.AddParameterBlock(bai, 6);
        problem.AddParameterBlock(pj, 7, new PoseLocalParameterization());
        problem.AddParameterBlock(vj, 3);
        problem.AddParameterBlock(baj, 6);
        problem.AddResidualBlock(
            new PreintFactor(imuPreint.get()),
            nullptr, pi, vi, bai, pj, vj, baj);

        ceres::Solver::Options opts;
        opts.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary sum;
        ceres::Solve(opts, &problem, &sum);

        finalPose.pos.x() = pj[0];
        finalPose.pos.y() = pj[1];
        finalPose.pos.z() = pj[2];
        finalPose.rot = Eigen::Quaterniond(
                            pj[6], pj[3], pj[4], pj[5])
                            .normalized();
      }

      // 9) Publish
      {
        geometry_msgs::PoseStamped ps;
        ps.header.stamp = cloudTime;
        ps.header.frame_id = "map";
        ps.pose.position.x = finalPose.pos.x();
        ps.pose.position.y = finalPose.pos.y();
        ps.pose.position.z = finalPose.pos.z();
        ps.pose.orientation.x = finalPose.rot.x();
        ps.pose.orientation.y = finalPose.rot.y();
        ps.pose.orientation.z = finalPose.rot.z();
        ps.pose.orientation.w = finalPose.rot.w();
        relocPub.publish(ps);

        //--- extract final RPY
        double roll = finalPose.roll();
        double pitch = finalPose.pitch();
        double yaw = finalPose.yaw();

        //--- extract pre‑pose (the prevPose at entry to this iteration)
        double pre_x = prevPose.pos.x();
        double pre_y = prevPose.pos.y();
        double pre_z = prevPose.pos.z();
        double pre_roll = prevPose.roll();
        double pre_pitch = prevPose.pitch();
        double pre_yaw = prevPose.yaw();

        //--- the exact timestamp you just published
        double pub_ts = ps.header.stamp.toSec();

        //--- print everything in green
        ROS_INFO(
            "\033[1;32m[LITELOAM %d]%s pub_ts=%.3f\n"
            "    pre_pos(%.3f, %.3f, %.3f), pre_RPY(%.1f, %.1f, %.1f)\n"
            "    final_pos(%.3f, %.3f, %.3f), final_RPY(%.1f, %.1f, %.1f)\033[0m",
            liteloam_id,
            printFirst ? " ** FIRST CLOUD **" : "",
            pub_ts,
            pre_x, pre_y, pre_z, pre_roll, pre_pitch, pre_yaw,
            ps.pose.position.x,
            ps.pose.position.y,
            ps.pose.position.z,
            roll, pitch, yaw);

        sensor_msgs::PointCloud2 outMsg;
        pcl::PointCloud<PointXYZI> aligned;
        pcl::transformPointCloud(
            *srcFine, aligned, finalPose.tfMat().cast<float>());
        pcl::toROSMsg(aligned, outMsg);
        outMsg.header.stamp = cloudTime;
        outMsg.header.frame_id = "map";
        alignedCloudPub.publish(outMsg);
      }

      // 10) update
      imuPreint.reset(nullptr);
      prevPose = finalPose;
      converged_ = true;
    }

    ROS_INFO("LITELOAM %d exiting", liteloam_id);
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

  // Buffer to store recent point clouds and their timestamps
  struct TimedCloud
  {
    CloudXYZIPtr cloud;
    ros::Time stamp;
  };
  std::deque<TimedCloud> cloud_queue;
  std::mutex cloud_mutex;

  // Buffer to store recent UWB pose messages
  std::deque<geometry_msgs::PoseStamped::ConstPtr> uloc_buffer;
  std::mutex uloc_mutex;

  // Publisher
  ros::Publisher relocPub;

  CloudXYZIPtr priorMap;
  KdFLANNPtr kdTreeMap;
  bool priorMapReady = false;

  std::mutex loam_mtx;
  std::vector<std::shared_ptr<LITELOAM>> loamInstances;
  std::thread checkLoamThread;

  std::thread bufferWatcherThread;
  const double BUFFER_TIMEOUT_SEC = 30.0;

public:
  ~Relocalization()
  {
    if (checkLoamThread.joinable())
      checkLoamThread.join();
    if (bufferWatcherThread.joinable())
      bufferWatcherThread.join();
  }

  Relocalization(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
  {
    Initialize();
    checkLoamThread = std::thread(&Relocalization::CheckLiteLoams, this);
  }

  void CheckLiteLoams()
  {
    while (ros::ok())
    {
      {
        std::lock_guard<std::mutex> lg(loam_mtx);

        // Iterate through all LITELOAM instances
        for (size_t lidx = 0; lidx < loamInstances.size(); ++lidx)
        {
          auto &loam = loamInstances[lidx];
          if (!loam)
            continue;

          // If one instance has been running too long but not converged,
          // restart
          if (loam->timeSinceStart() > 12.0 && !loam->loamConverged() &&
              loam->isRunning())
          {
            ROS_INFO("[Relocalization] LITELOAM %d exceeded 12s. Restart...",
                     loam->getID());
            loam->stop();
          }
          else if (loam->loamConverged())
          {
            // do something else
          }
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }

  void Initialize()
  {
    // ULOC pose subcriber
    ulocSub =
        nh_ptr->subscribe("/uwb_pose", 10, &Relocalization::ULOCCallback, this);

    lidarCloudSub = nh_ptr->subscribe("/lastcloud", 100, &Relocalization::PCHandler, this);

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

  void PCHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    CloudXYZIPtr cloud(new CloudXYZI());
    pcl::fromROSMsg(*msg, *cloud);

    std::lock_guard<std::mutex> lock(cloud_mutex);
    cloud_queue.push_back({cloud, msg->header.stamp});
    // Limit buffer size to avoid unbounded growth
    if (cloud_queue.size() > 200)
      cloud_queue.pop_front();
  }

  // Match UWB pose timestamps to the nearest LiDAR cloud, then spawn/restart LITELOAM
  void ULOCCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
  {
    if (!priorMapReady)
    {
      ROS_WARN("Prior map not yet loaded; skipping UWB callback");
      return;
    }

    // Buffer the incoming UWB pose
    {
      std::lock_guard<std::mutex> lock(uloc_mutex);
      uloc_buffer.push_back(msg);
      if (uloc_buffer.size() > 50)
        uloc_buffer.pop_front();
    }

    // Get the timestamp of the most recent LiDAR cloud
    ros::Time latest_cloud_time;
    {
      std::lock_guard<std::mutex> lock(cloud_mutex);
      if (cloud_queue.empty())
      {
        ROS_WARN("No LiDAR clouds received yet; cannot match timestamp");
        return;
      }
      latest_cloud_time = cloud_queue.back().stamp;
    }

    // 1) Try to find a UWB pose newer than the cloud (dt >= 0)
    //    within a 5s window, whose dt is closest to 3s
    geometry_msgs::PoseStamped::ConstPtr best_pose = nullptr;
    {
      std::lock_guard<std::mutex> lock(uloc_mutex);
      const double target_offset = 3.0; // we want ~3 seconds after cloud
      const double max_window = 5.0;    // only consider up to +5s
      double best_err = std::numeric_limits<double>::infinity();

      for (auto &pose_msg : uloc_buffer)
      {
        double dt = (pose_msg->header.stamp - latest_cloud_time).toSec();
        // only consider poses newer than cloud and not beyond our window
        if (dt >= 0.0 && dt <= max_window)
        {
          double err = std::abs(dt - target_offset);
          if (err < best_err)
          {
            best_err = err;
            best_pose = pose_msg;
          }
        }
      }

      // 2) Fallback: if none in [0,5s], pick the pose with the closest timestamp
      if (!best_pose)
      {
        double best_diff = std::numeric_limits<double>::infinity();
        for (auto &pose_msg : uloc_buffer)
        {
          double diff = std::abs(
              (pose_msg->header.stamp - latest_cloud_time).toSec());
          if (diff < best_diff)
          {
            best_diff = diff;
            best_pose = pose_msg;
          }
        }
      }
    }

    if (!best_pose)
    {
      ROS_WARN("Could not find a matching UWB pose for cloud at %f",
               latest_cloud_time.toSec());
      return;
    }

    // Wrap the matched pose into your transform type
    mytf start_pose(*best_pose);

    // Spawn or restart a LITELOAM instance with that start pose
    {
      std::lock_guard<std::mutex> lg(loam_mtx);

      // If fewer than x instances exist, create a fresh one
      if (loamInstances.size() < 5)
      {
        int newID = loamInstances.size();
        auto inst = std::make_shared<LITELOAM>(
            priorMap, kdTreeMap, start_pose, newID, nh_ptr);
        loamInstances.push_back(inst);
        ROS_INFO("Created LITELOAM ID=%d (matched at %f)",
                 newID, best_pose->header.stamp.toSec());
      }
      else
      {
        // Otherwise restart the first non-running instance
        for (auto &loam : loamInstances)
        {
          if (!loam->isRunning())
          {
            int id = loam->getID();
            loam = std::make_shared<LITELOAM>(
                priorMap, kdTreeMap, start_pose, id, nh_ptr);
            ROS_INFO("Restarted LITELOAM ID=%d (matched at %f)",
                     id, best_pose->header.stamp.toSec());
            break;
          }
        }
      }
    }
  }

  void BufferWatcher()
  {
    ros::Rate rate(1.0);
    while (ros::ok())
    {
      ros::Time now = ros::Time::now();
      ros::Time latest_cloud = ros::Time(0);
      ros::Time latest_uloc = ros::Time(0);

      {
        std::lock_guard<std::mutex> lock(cloud_mutex);
        if (!cloud_queue.empty())
          latest_cloud = cloud_queue.back().stamp;
      }
      {
        std::lock_guard<std::mutex> lock(uloc_mutex);
        if (!uloc_buffer.empty())
          latest_uloc = uloc_buffer.back()->header.stamp;
      }

      double dt_cloud = (now - latest_cloud).toSec();
      double dt_uloc = (now - latest_uloc).toSec();

      if (dt_cloud > BUFFER_TIMEOUT_SEC || dt_uloc > BUFFER_TIMEOUT_SEC)
      {
        ROS_ERROR("[Relocalization] No data for %.1f s (cloud: %.1f s, uloc: %.1f s). Shutting down.",
                  BUFFER_TIMEOUT_SEC, dt_cloud, dt_uloc);
        ros::shutdown();
        return;
      }

      rate.sleep();
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