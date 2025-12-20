/**
 * RELOCALIZATION NODE
 * * Description:
 * Multi-hypothesis relocalization system using LiDAR, IMU, and a Seed Pose (UWB/GPS/Rviz).
 * It manages multiple LITELOAM instances to find the best match against a Prior Map.
 */

// ==========================================
// INCLUDES
// ==========================================
#include "STDesc.h"
#include "slict/FeatureCloud.h"
#include "utility.h"
#include <optional>
#include <thread>
#include <mutex>
#include <atomic>
#include <deque>

// ROS
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>

// Optimization & Math
#include "PoseLocalParameterization.h"
#include <ceres/ceres.h>
#include <Eigen/Dense>

// PCL
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

// Project Specific
#include "PreintBase.h"
#include <factor/PreintFactor.h>

// ==========================================
// DEFINES & CONSTANTS
// ==========================================
#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"
#define RESET "\033[0m"

// Type Definitions
typedef sensor_msgs::PointCloud2::ConstPtr rosCloudMsgPtr;
typedef sensor_msgs::PointCloud2 rosCloudMsg;

struct TimedCloud
{
  CloudXYZIPtr cloud;
  ros::Time stamp;
};

// Helper function for timing
static inline double to_ms(const std::chrono::steady_clock::time_point &t0,
                           const std::chrono::steady_clock::time_point &t1)
{
  using namespace std::chrono;
  return duration_cast<duration<double, std::milli>>(t1 - t0).count();
}

// ==========================================
// CLASS: LITELOAM (Worker)
// ==========================================
class LITELOAM
{
private:
  ros::NodeHandlePtr nh_ptr;
  ros::Subscriber lidarCloudSub;
  ros::Publisher relocPub;

  // Data Buffers
  std::deque<TimedCloud> cloud_queue;
  std::mutex cloud_mutex;

  // Shared IMU Data
  const std::deque<sensor_msgs::ImuConstPtr> *imu_queue_shared = nullptr;
  std::mutex *imu_mutex_shared = nullptr;
  double last_imu_used_time = -1.0;

  // Core Data
  CloudXYZIPtr priorMap;
  std::string lidar_topic_ = "/lastcloud";
  int liteloam_id;

  // Threading
  std::thread processThread;
  std::atomic<bool> running{true};
  double startTime = 0.0;
  double last_cloud_time = -1.0;

  // States
  mytf initPose;
  mytf prevPose;

  // IMU Pre-integration
  std::unique_ptr<PreintBase> imuPreint;
  Eigen::Vector3d v_prev = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_curr = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();

  // IMU Noise Params
  double acc_n = 0.6, acc_w = 0.08, gyr_n = 0.05, gyr_w = 0.003;
  Eigen::Vector3d gvec = Eigen::Vector3d(0, 0, 9.82);

  // ICP Parameters
  int iterCoarse = 10;
  int iterFine = 15;
  double transEps = 1e-6;
  double fitEps = 1e-5;
  double ransacThresh = 0.1;
  bool use_recip_corr = false;
  double maxcorrCoarse = 10.0;
  double maxcorrFine = 3.0;
  float leafCoarse = 0.20f;
  float leafFine = 0.08f;
  double icpThresh = 2.0;
  double yaw_step_first = 45.0;

  // --- ADAPTIVE DOWNSAMPLING PARAMS -------
  int max_lidar_factor = 6000;
  double ds_radius = 0.4;
  double min_leaf_size = 0.1;

  double last_adaptive_time_ = -1.0;
  int adaptive_scale_ = 0;

  // Flags
  bool fine_enable = false;
  bool icpConverged = false;
  bool posePublished = false;
  bool firstCloud = true;
  bool icpOnly = true;
  bool terminate_after_publish = true;

public:
  LITELOAM(const CloudXYZIPtr &priorMap,
           const mytf &initPose,
           int liteloam_id,
           const ros::NodeHandlePtr &nh_ptr,
           const std::optional<TimedCloud> &seed_cloud,
           const std::deque<sensor_msgs::ImuConstPtr> *imu_q_shared,
           std::mutex *imu_mtx_shared)
      : nh_ptr(nh_ptr),
        priorMap(priorMap),
        initPose(initPose),
        prevPose(initPose),
        liteloam_id(liteloam_id)
  {
    ros::NodeHandle pnh("~");

    // Load ICP Parameters
    pnh.param("iterCoarse", iterCoarse, iterCoarse);
    pnh.param("iterFine", iterFine, iterFine);
    pnh.param("maxcorrCoarse", maxcorrCoarse, maxcorrCoarse);
    pnh.param("maxcorrFine", maxcorrFine, maxcorrFine);
    pnh.param("leafCoarse", leafCoarse, leafCoarse);
    pnh.param("leafFine", leafFine, leafFine);

    pnh.param("max_lidar_factor", max_lidar_factor, 6000);
    pnh.param("ds_radius", ds_radius, 0.4);
    pnh.param("min_leaf_size", min_leaf_size, 0.1);

    pnh.param("icpThresh", icpThresh, icpThresh);
    pnh.param("icpOnly", icpOnly, true);

    pnh.param("terminate_after_publish", terminate_after_publish, true);
    pnh.param("yaw_step_first", yaw_step_first, yaw_step_first);
    pnh.param("fine_enable", fine_enable, fine_enable);
    pnh.param<std::string>("lidar_topic", lidar_topic_, "/lastcloud");

    // Load Gravity Config
    double gz_param = 9.82;
    pnh.param("gravity_z", gz_param, gz_param);
    this->gvec = Eigen::Vector3d(0.0, 0.0, gz_param);

    // Initialize Cloud Queue
    if (seed_cloud.has_value())
    {
      std::lock_guard<std::mutex> lk(cloud_mutex);
      cloud_queue.push_back(*seed_cloud);
    }

    // Shared IMU Reference
    imu_queue_shared = imu_q_shared;
    imu_mutex_shared = imu_mtx_shared;

    // Subscribers & Publishers
    lidarCloudSub = nh_ptr->subscribe(lidar_topic_, 100, &LITELOAM::PCHandler, this);
    relocPub = nh_ptr->advertise<geometry_msgs::PoseStamped>("/liteloam_pose", 100);

    // Start Processing Thread
    startTime = ros::Time::now().toSec();
    running = true;
    processThread = std::thread(&LITELOAM::processBuffer, this);

    printf(KBLU "[LITELOAM %d] Constructed. InitPose: [%.2f, %.2f, %.2f]\n" RESET,
           liteloam_id, initPose.pos.x(), initPose.pos.y(), initPose.pos.z());
  }

  ~LITELOAM()
  {
    shutdown();
    printf(KYEL "[LITELOAM %d] Destructed.\n" RESET, liteloam_id);
  }

  void stop() { running = false; }
  double timeSinceStart() const { return ros::Time::now().toSec() - startTime; }
  bool loamConverged() const { return icpConverged; }
  bool published() const { return posePublished; }
  bool isRunning() const { return running; }
  int getID() const { return liteloam_id; }

  void shutdown()
  {
    running = false;
    if (processThread.joinable())
      processThread.join();

    {
      std::lock_guard<std::mutex> lk(cloud_mutex);
      cloud_queue.clear();
    }
    lidarCloudSub.shutdown();
    relocPub.shutdown();
  }

private:
  void PCHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    CloudXYZIPtr cloud(new CloudXYZI());
    pcl::fromROSMsg(*msg, *cloud);

    std::lock_guard<std::mutex> lk(cloud_mutex);
    cloud_queue.push_back({cloud, msg->header.stamp});

    if (cloud_queue.size() > 5)
      cloud_queue.pop_front();
  }

  void processBuffer()
  {
    while (running && ros::ok())
    {
      // 1. Wait for PointCloud
      if (cloud_queue.empty())
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      // 2. Pop PointCloud
      TimedCloud item;
      {
        std::lock_guard<std::mutex> lk(cloud_mutex);
        item = cloud_queue.front();
        cloud_queue.pop_front();
      }

      auto t_frame_start = std::chrono::steady_clock::now();
      CloudXYZIPtr raw = item.cloud;
      double tNow = item.stamp.toSec();
      double tPrev = (last_cloud_time < 0.0 ? tNow : last_cloud_time);
      last_cloud_time = tNow;

      // 3. Downsample (Coarse & Fine)
      CloudXYZIPtr srcCoarse(new CloudXYZI()), srcFine(new CloudXYZI());
      {
        // --- LOGIC ADAPTIVE DOWNSAMPLING CHO COARSE CLOUD ---
        // 1. Reset scale 
        if (last_adaptive_time_ < 0 || (tNow - last_adaptive_time_) > 5.0)
        {
          last_adaptive_time_ = tNow;
          adaptive_scale_ = 0;
        }

        // 2. Adaptive Loop
        if (ds_radius > 0.0)
        {
          int current_scale = adaptive_scale_;
          pcl::UniformSampling<PointXYZI> downsampler;
          while (true)
          {
            double ds_effective_radius = ds_radius / (std::pow(2, current_scale));

            if (ds_effective_radius < min_leaf_size)
              ds_effective_radius = min_leaf_size;

            downsampler.setInputCloud(raw);
            downsampler.setRadiusSearch(ds_effective_radius);
            downsampler.filter(*srcCoarse);

            if (srcCoarse->size() >= max_lidar_factor ||
                srcCoarse->size() == raw->size() ||
                ds_effective_radius <= min_leaf_size)
            {
              break;
            }
            else
            {
              // printf(KYEL "[LITELOAM %d] Too few points (%zu). Radius %.3f -> Relaxing...\n" RESET,
              //        liteloam_id, srcCoarse->size(), ds_effective_radius);
              current_scale++;

              if (ds_effective_radius <= min_leaf_size)
                break;
            }
          }


          if (current_scale != adaptive_scale_)
          {
            adaptive_scale_ = current_scale;
            last_adaptive_time_ = tNow;
          }
        }
        else
        {
          pcl::VoxelGrid<PointXYZI> vg;
          vg.setInputCloud(raw);
          vg.setLeafSize(leafCoarse, leafCoarse, leafCoarse);
          vg.filter(*srcCoarse);
        }

        // --- FINE CLOUD ----
        pcl::VoxelGrid<PointXYZI> vgFine;
        vgFine.setInputCloud(raw);
        vgFine.setLeafSize(leafFine, leafFine, leafFine);
        vgFine.filter(*srcFine);
      }

      // 4. Collect IMU Data
      std::vector<sensor_msgs::ImuConstPtr> imuBuf;
      double used_max_t = -1.0;
      {
        std::lock_guard<std::mutex> lk(*imu_mutex_shared);
        const double tStart = std::max(tPrev, last_imu_used_time);

        for (const auto &p : *imu_queue_shared)
        {
          const double t = p->header.stamp.toSec();
          if (t <= tStart)
            continue;
          if (t > tNow)
            break;
          imuBuf.push_back(p);
          if (t > used_max_t)
            used_max_t = t;
        }
      }
      if (used_max_t > last_imu_used_time)
        last_imu_used_time = used_max_t;

      // 5. Perform IMU Pre-integration
      if (!imuBuf.empty())
      {
        Eigen::Vector3d a0(imuBuf.front()->linear_acceleration.x, imuBuf.front()->linear_acceleration.y, imuBuf.front()->linear_acceleration.z);
        Eigen::Vector3d g0(imuBuf.front()->angular_velocity.x, imuBuf.front()->angular_velocity.y, imuBuf.front()->angular_velocity.z);

        if (!imuPreint)
        {
          imuPreint.reset(new PreintBase(a0, g0, ba, bg, true, acc_n, acc_w, gyr_n, gyr_w, gvec, liteloam_id));
        }
        else
        {
          imuPreint->repropagate(ba, bg);
        }

        for (size_t i = 1; i < imuBuf.size(); ++i)
        {
          double dti = imuBuf[i]->header.stamp.toSec() - imuBuf[i - 1]->header.stamp.toSec();
          Eigen::Vector3d ai(imuBuf[i]->linear_acceleration.x, imuBuf[i]->linear_acceleration.y, imuBuf[i]->linear_acceleration.z);
          Eigen::Vector3d gi(imuBuf[i]->angular_velocity.x, imuBuf[i]->angular_velocity.y, imuBuf[i]->angular_velocity.z);
          imuPreint->push_back(dti, ai, gi);
        }
      }
      else
      {
        imuPreint.reset(nullptr);
      }

      // 6. Predict Initial Pose (Dead Reckoning)
      mytf predPose = prevPose;
      if (imuPreint)
      {
        Eigen::Vector3d dp, dv;
        Eigen::Quaterniond dq;
        imuPreint->getPredictedDeltas(dp, dv, dq, prevPose.rot, Eigen::Vector3d::Zero(), imuPreint->linearized_ba, imuPreint->linearized_bg);
        predPose.pos = prevPose.pos + (prevPose.rot * dp);
        predPose.rot = (prevPose.rot * dq).normalized();
      }

      // 7. Run ICP (Coarse -> Fine)
      Eigen::Matrix4f bestTrans = Eigen::Matrix4f::Identity();
      double bestFitness = std::numeric_limits<double>::infinity();

      Eigen::Quaternionf q0 = predPose.rot.cast<float>().normalized();
      Eigen::Matrix3f R0 = q0.toRotationMatrix();
      Eigen::Vector3f T0 = predPose.pos.cast<float>();

      // Setup Yaw Search Steps
      std::vector<double> yawOffsets;
      if (firstCloud)
      {
        for (double d = -180.0; d <= 180.0; d += yaw_step_first)
          yawOffsets.push_back(d);
      }
      else
      {
        yawOffsets.push_back(0.0);
      }

      // ICP Helper Lambda
      auto run_icp = [&](const CloudXYZIPtr &src, double max_corr, int max_iter, const Eigen::Matrix4f &init, Eigen::Matrix4f &T_out, double &fitness) -> bool
      {
        pcl::IterativeClosestPoint<PointXYZI, PointXYZI> icp;
        icp.setInputSource(src);
        icp.setInputTarget(priorMap);
        icp.setUseReciprocalCorrespondences(use_recip_corr);
        icp.setMaxCorrespondenceDistance(max_corr);
        icp.setMaximumIterations(max_iter);
        icp.setTransformationEpsilon(transEps);
        icp.setEuclideanFitnessEpsilon(fitEps);
        icp.setRANSACOutlierRejectionThreshold(ransacThresh);

        pcl::PointCloud<PointXYZI> aligned;
        icp.align(aligned, init);
        if (!icp.hasConverged())
          return false;

        fitness = icp.getFitnessScore();
        T_out = icp.getFinalTransformation();
        return true;
      };

      // 7a. Coarse ICP
      Eigen::Matrix4f coarse_best_T = Eigen::Matrix4f::Identity();
      double coarse_best_fit = std::numeric_limits<double>::infinity();

      for (double d : yawOffsets)
      {
        double yawRad = d * M_PI / 180.0;
        Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
        guess.block<3, 3>(0, 0) = (Eigen::AngleAxisf(yawRad, Eigen::Vector3f::UnitZ()) * R0).matrix();
        guess.block<3, 1>(0, 3) = T0;

        Eigen::Matrix4f T_tmp;
        double fit_tmp;
        if (run_icp(srcCoarse, maxcorrCoarse, iterCoarse, guess, T_tmp, fit_tmp))
        {
          if (fit_tmp < coarse_best_fit)
          {
            coarse_best_fit = fit_tmp;
            coarse_best_T = T_tmp;
          }
        }
      }

      if (!std::isfinite(coarse_best_fit))
      {
        printf(KYEL "[LITELOAM %d] FAILED (Coarse ICP). Stamp: %.3f. Init: [%.2f, %.2f, %.2f]\n" RESET,
               liteloam_id, tNow, predPose.pos.x(), predPose.pos.y(), predPose.pos.z());
        icpConverged = false;
        running = false;
        continue;
      }

      bestTrans = coarse_best_T;
      bestFitness = coarse_best_fit;

      // 7b. Fine ICP
      if (fine_enable)
      {
        Eigen::Matrix4f T_fine;
        double fit_fine;
        if (!run_icp(srcFine, maxcorrFine, iterFine, coarse_best_T, T_fine, fit_fine))
        {
          printf(KYEL "[LITELOAM %d] FAILED (Fine ICP). Stamp: %.3f. Init: [%.2f, %.2f, %.2f]\n" RESET,
                 liteloam_id, tNow, predPose.pos.x(), predPose.pos.y(), predPose.pos.z());
          icpConverged = false;
          running = false;
          continue;
        }
        bestTrans = T_fine;
        bestFitness = fit_fine;
      }

      if (bestFitness >= icpThresh)
      {
        printf(KYEL "[LITELOAM %d] FAILED (Fitness %.4f > %.1f). Stamp: %.3f. Init: [%.2f, %.2f, %.2f]\n" RESET,
               liteloam_id, bestFitness, icpThresh, tNow, predPose.pos.x(), predPose.pos.y(), predPose.pos.z());
        icpConverged = false;
        running = false;
        continue;
      }

      // 8. Optimize with IMU Factor (Ceres Solver)
      Eigen::Matrix4d bestTransD = bestTrans.cast<double>();
      mytf finalPose(bestTransD);
      if (finalPose.rot.w() < 0.0)
        finalPose.rot.coeffs() *= -1.0;

      if (imuPreint && !icpOnly)
      {
        double pi[7] = {prevPose.pos.x(), prevPose.pos.y(), prevPose.pos.z(), prevPose.rot.x(), prevPose.rot.y(), prevPose.rot.z(), prevPose.rot.w()};
        double pj[7] = {finalPose.pos.x(), finalPose.pos.y(), finalPose.pos.z(), finalPose.rot.x(), finalPose.rot.y(), finalPose.rot.z(), finalPose.rot.w()};
        double vi[3] = {v_prev.x(), v_prev.y(), v_prev.z()};
        double vj[3] = {v_curr.x(), v_curr.y(), v_curr.z()};
        double bai[6] = {ba.x(), ba.y(), ba.z(), bg.x(), bg.y(), bg.z()};
        double baj[6] = {ba.x(), ba.y(), ba.z(), bg.x(), bg.y(), bg.z()};

        ceres::Problem problem;
        auto *pose_plus = new PoseLocalParameterization();

        problem.AddParameterBlock(pi, 7, pose_plus);
        problem.AddParameterBlock(pj, 7, pose_plus);
        problem.AddParameterBlock(vi, 3);
        problem.AddParameterBlock(vj, 3);
        problem.AddParameterBlock(bai, 6);
        problem.AddParameterBlock(baj, 6);

        problem.SetParameterBlockConstant(pi);
        problem.SetParameterBlockConstant(vi);
        problem.SetParameterBlockConstant(bai);

        // Set bounds for Bias
        for (int k = 0; k < 3; ++k)
        {
          problem.SetParameterLowerBound(baj, k, -0.5);
          problem.SetParameterUpperBound(baj, k, 0.5);
          problem.SetParameterLowerBound(baj, 3 + k, -0.1);
          problem.SetParameterUpperBound(baj, 3 + k, 0.1);
        }

        // Add IMU Residual
        ceres::CostFunction *f_preint = new PreintFactor(imuPreint.get());
        problem.AddResidualBlock(f_preint, nullptr, pi, vi, bai, pj, vj, baj);

        ceres::Solver::Options opts;
        opts.max_num_iterations = 20;
        opts.max_solver_time_in_seconds = 0.02;
        opts.linear_solver_type = ceres::DENSE_QR;
        opts.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary sum;
        ceres::Solve(opts, &problem, &sum);

        // Update State
        finalPose.pos.x() = pj[0];
        finalPose.pos.y() = pj[1];
        finalPose.pos.z() = pj[2];
        finalPose.rot = Eigen::Quaterniond(pj[6], pj[3], pj[4], pj[5]).normalized();
        v_curr = Eigen::Vector3d(vj[0], vj[1], vj[2]);
        ba = Eigen::Vector3d(baj[0], baj[1], baj[2]);
        bg = Eigen::Vector3d(baj[3], baj[4], baj[5]);
      }

      bool ready_to_publish = true;
      if (!icpOnly && firstCloud)
      {
        ready_to_publish = false;
      }

      if (ready_to_publish)
      {
        // 9. Publish Result
        double proc_ms = to_ms(t_frame_start, std::chrono::steady_clock::now());
        ros::Time stamp_out = item.stamp + ros::Duration(proc_ms / 1000.0);

        geometry_msgs::PoseStamped ps;
        ps.header.stamp = stamp_out;
        ps.header.frame_id = "map";
        ps.pose.position.x = finalPose.pos.x();
        ps.pose.position.y = finalPose.pos.y();
        ps.pose.position.z = finalPose.pos.z();
        ps.pose.orientation.x = finalPose.rot.x();
        ps.pose.orientation.y = finalPose.rot.y();
        ps.pose.orientation.z = finalPose.rot.z();
        ps.pose.orientation.w = finalPose.rot.w();
        relocPub.publish(ps);

        // LOGGING: SUCCESS
        printf(KGRN
               "[LITELOAM %d] SUCCESS | Fit: %.4f | IMU: %zu | Frame: %s | Stamp: %.3f | Proc: %.1fms\n"
               "  > Init: [%.2f, %.2f, %.2f] YPR: [%.0f, %.0f, %.0f]\n"
               "  > Final:[%.2f, %.2f, %.2f] YPR: [%.0f, %.0f, %.0f]\n" RESET,
               liteloam_id,
               bestFitness,
               imuBuf.size(),
               firstCloud ? "1st" : "2nd+", // Log rõ là frame mấy
               stamp_out.toSec(),
               proc_ms,
               predPose.pos.x(), predPose.pos.y(), predPose.pos.z(),
               predPose.yaw(), predPose.pitch(), predPose.roll(),
               finalPose.pos.x(), finalPose.pos.y(), finalPose.pos.z(),
               finalPose.yaw(), finalPose.pitch(), finalPose.roll());

        posePublished = true;

        if (terminate_after_publish)
          running = false;
      }
      else
      {
        printf(KYEL "[LITELOAM %d] IMU Mode: Init frame processed. Waiting for 2nd frame to verify...\n" RESET, liteloam_id);
      }

      // 10. Update State
      firstCloud = false;
      icpConverged = true;
      imuPreint.reset(nullptr);
      v_prev = v_curr;
      prevPose = finalPose;
    }

    printf("[LITELOAM %d] Exiting processing thread.\n", liteloam_id);
  }
};

// ==========================================
// CLASS: Relocalization (Manager)
// ==========================================
class Relocalization
{
private:
  ros::NodeHandlePtr nh_ptr;

  // ROS IO
  ros::Subscriber lidarCloudSub;
  ros::Subscriber seedSub;
  ros::Subscriber imuSub;
  ros::Publisher relocPub;

  // Config Params
  std::string lidar_topic_ = "/lastcloud";
  std::string seed_topic_ = "/seed_pose";
  std::string imu_topic_ = "/vn100/imu";
  int liteloamNum = 5;
  bool shutdown_on_success = true;

  // IMU Logic
  int imu_type_ = 2; // 0: Scale (input g), 1: Flip (input m/s2 but inverted), 2: Normal
  int imu_max_keep = 1200;
  double imu_max_age = 20.0;

  // Data Storage
  std::deque<TimedCloud> cloud_queue;
  std::mutex cloud_mutex;

  std::deque<geometry_msgs::PoseStamped::ConstPtr> seed_buffer;
  std::mutex seed_mutex;

  std::deque<sensor_msgs::ImuConstPtr> imu_queue_shared;
  std::mutex imu_mutex_shared;

  // Map
  CloudXYZIPtr priorMap;
  bool priorMapReady = false;

  // Workers Management
  std::mutex loam_mtx;
  std::vector<std::shared_ptr<LITELOAM>> loamInstances;
  std::thread checkLoamThread;
  std::atomic<bool> active_{true};

public:
  Relocalization(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
  {
    Initialize();
    checkLoamThread = std::thread(&Relocalization::CheckLiteLoams, this);
  }

  ~Relocalization()
  {
    if (checkLoamThread.joinable())
      checkLoamThread.join();
  }

private:
  void Initialize()
  {
    ros::NodeHandle pnh("~");

    // Load Topic Params
    pnh.param<std::string>("lidar_topic", lidar_topic_, lidar_topic_);
    pnh.param<std::string>("seed_pose_topic", seed_topic_, seed_topic_);
    pnh.param<std::string>("imu_topic", imu_topic_, imu_topic_);

    // Load IMU Config
    pnh.param("imu_type", imu_type_, 2);
    pnh.param("imu_max_keep", imu_max_keep, imu_max_keep);
    pnh.param("imu_max_age", imu_max_age, imu_max_age);
    ROS_INFO("[RELOC] IMU Config Type: %d", imu_type_);

    // Load Worker Config
    pnh.param("liteloamNum", liteloamNum, liteloamNum);
    pnh.param("shutdown_on_success", shutdown_on_success, true);

    // Load Prior Map
    std::string prior_map_dir = "";
    pnh.param("priormap_dir", prior_map_dir, std::string(""));
    ROS_INFO("[RELOC] Map Dir: %s", prior_map_dir.c_str());

    priorMap.reset(new CloudXYZI());
    std::string pcd_file = prior_map_dir + "/priormap.pcd";

    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *priorMap) != 0)
    {
      ROS_ERROR("[RELOC] Failed to load PCD: %s", pcd_file.c_str());
      return;
    }

    ROS_INFO("[RELOC] Loaded Prior Map (%zu points). Downsampling...", priorMap->size());

    // Downsample Map for Performance
    pcl::UniformSampling<pcl::PointXYZI> downsampler;
    downsampler.setInputCloud(priorMap);
    downsampler.setRadiusSearch(1.0);
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampledMap(new pcl::PointCloud<pcl::PointXYZI>());
    downsampler.filter(*downsampledMap);
    priorMap = downsampledMap;

    ROS_INFO("[RELOC] Downsampled Map (%zu points). Ready.", priorMap->size());
    priorMapReady = true;

    // Subscribers
    seedSub = nh_ptr->subscribe(seed_topic_, 100, &Relocalization::SeedPoseCallback, this);
    lidarCloudSub = nh_ptr->subscribe(lidar_topic_, 100, &Relocalization::PCHandler, this);
    imuSub = nh_ptr->subscribe(imu_topic_, 1000, &Relocalization::IMUHandler, this);

    ROS_INFO(KGRN "----> Relocalization Node Initialized." RESET);
  }

  void CheckLiteLoams()
  {
    while (ros::ok())
    {
      bool should_quit = false;
      {
        std::lock_guard<std::mutex> lg(loam_mtx);
        for (auto &loam : loamInstances)
        {
          if (!loam)
            continue;

          // Kill if timeout or failed
          if (loam->timeSinceStart() > 10.0 && !loam->loamConverged() && loam->isRunning())
          {
            printf("[Relocalization] LITELOAM %d Timeout (>10s). Stopping.\n", loam->getID());
            loam->stop();
          }
          // Success condition
          else if (loam->published() && shutdown_on_success)
          {
            should_quit = true;
          }
        }
      }

      if (should_quit)
      {
        ROS_INFO("[Relocalization] Success! Shutting down all workers...");

        std::vector<std::shared_ptr<LITELOAM>> locals;
        {
          std::lock_guard<std::mutex> lg(loam_mtx);
          locals.swap(loamInstances);
        }

        for (auto &lm : locals)
          if (lm)
            lm->shutdown();

        lidarCloudSub.shutdown();
        seedSub.shutdown();
        imuSub.shutdown();
        active_.store(false);

        ros::requestShutdown();
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  void PCHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    CloudXYZIPtr cloud(new CloudXYZI());
    pcl::fromROSMsg(*msg, *cloud);

    std::lock_guard<std::mutex> lock(cloud_mutex);
    cloud_queue.push_back({cloud, msg->header.stamp});
    if (cloud_queue.size() > 10)
      cloud_queue.pop_front();
  }

  void IMUHandler(const sensor_msgs::ImuConstPtr &msg)
  {
    sensor_msgs::ImuPtr m(new sensor_msgs::Imu(*msg));
    if (m->header.stamp.isZero())
      m->header.stamp = ros::Time::now();

    double ax = m->linear_acceleration.x;
    double ay = m->linear_acceleration.y;
    double az = m->linear_acceleration.z;

    double gx = m->angular_velocity.x;
    double gy = m->angular_velocity.y;
    double gz = m->angular_velocity.z;

    // ==========================================
    // PROCESS IMU DATA BASED ON TYPE
    // ==========================================
    if (imu_type_ == 0) // CASE 0: Input is in 'g' units, correct orientation
    {
      double G_SCALE = 9.81;
      ax *= G_SCALE;
      ay *= G_SCALE;
      az *= G_SCALE;
    }
    else if (imu_type_ == 1) // CASE 1: Input is m/s^2, but mounted upside down (Flip YZ)
    {
      ax = ax;
      ay = -ay;
      az = -az;
      gx = gx;
      gy = -gy;
      gz = -gz;
    }
    else if (imu_type_ == 2) // CASE 2: Input is m/s^2, correct orientation (Standard)
    {
      // Pass-through
    }
    // ==========================================

    m->linear_acceleration.x = ax;
    m->linear_acceleration.y = ay;
    m->linear_acceleration.z = az;
    m->angular_velocity.x = gx;
    m->angular_velocity.y = gy;
    m->angular_velocity.z = gz;

    std::lock_guard<std::mutex> lk(imu_mutex_shared);
    imu_queue_shared.push_back(m);

    while (imu_queue_shared.size() > imu_max_keep)
      imu_queue_shared.pop_front();

    if (!imu_queue_shared.empty())
    {
      double t_now = imu_queue_shared.back()->header.stamp.toSec();
      while (!imu_queue_shared.empty() && (t_now - imu_queue_shared.front()->header.stamp.toSec()) > imu_max_age)
        imu_queue_shared.pop_front();
    }
  }

  void SeedPoseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
  {
    if (!priorMapReady)
    {
      ROS_WARN_THROTTLE(5.0, "[RELOC] Map not ready, ignoring Seed Pose.");
      return;
    }

    // Buffer Seed Pose
    {
      std::lock_guard<std::mutex> lock(seed_mutex);
      seed_buffer.push_back(msg);
      if (seed_buffer.size() > 1000)
        seed_buffer.pop_front();
    }

    // Get Latest LiDAR Cloud
    TimedCloud matched_cloud;
    {
      std::lock_guard<std::mutex> lock(cloud_mutex);
      if (cloud_queue.empty())
        return;
      matched_cloud = cloud_queue.back();
    }

    // Find Best Seed Pose Match (Time Synced)
    geometry_msgs::PoseStamped::ConstPtr best_pose = nullptr;
    double best_diff = std::numeric_limits<double>::infinity();
    ros::Time cloud_time = matched_cloud.stamp;

    {
      std::lock_guard<std::mutex> lock(seed_mutex);
      for (auto &pose_msg : seed_buffer)
      {
        double diff = std::abs((pose_msg->header.stamp - cloud_time).toSec());
        if (diff < best_diff)
        {
          best_diff = diff;
          best_pose = pose_msg;
        }
      }
    }

    if (!best_pose)
    {
      ROS_WARN("[RELOC] No matching Seed Pose found for cloud.");
      return;
    }

    // Remove used pose
    {
      std::lock_guard<std::mutex> lock(seed_mutex);
      auto it = std::find_if(seed_buffer.begin(), seed_buffer.end(),
                             [&](const geometry_msgs::PoseStamped::ConstPtr &p)
                             { return p.get() == best_pose.get(); });
      if (it != seed_buffer.end())
        seed_buffer.erase(it);
    }

    // Spawn Worker
    mytf start_pose(*best_pose);
    std::lock_guard<std::mutex> lg(loam_mtx);

    // Try to find available slot or restart finished one
    std::shared_ptr<LITELOAM> target_inst = nullptr;

    // 1. Try to reuse finished instance
    for (auto &loam : loamInstances)
    {
      if (!loam->isRunning())
      {
        target_inst = loam;
        break;
      }
    }

    if (target_inst)
    {
      // Restart existing
      int id = target_inst->getID();
      target_inst = std::make_shared<LITELOAM>(priorMap, start_pose, id, nh_ptr, std::make_optional(matched_cloud), &imu_queue_shared, &imu_mutex_shared);
      printf("[RELOC] Restarted LITELOAM %d with Seed Pose at %.3f\n", id, best_pose->header.stamp.toSec());

      // Re-assign pointer in vector
      for (auto &loam : loamInstances)
      {
        if (!loam->isRunning() && loam->getID() == id)
        {
          loam = target_inst;
          break;
        }
      }
    }
    else if (loamInstances.size() < liteloamNum)
    {
      // Create new
      int newID = static_cast<int>(loamInstances.size());
      auto inst = std::make_shared<LITELOAM>(priorMap, start_pose, newID, nh_ptr, std::make_optional(matched_cloud), &imu_queue_shared, &imu_mutex_shared);
      loamInstances.push_back(inst);
      printf("[RELOC] Created LITELOAM %d with Seed Pose at %.3f\n", newID, best_pose->header.stamp.toSec());
    }
  }
};

// ==========================================
// MAIN
// ==========================================
int main(int argc, char **argv)
{
  ros::init(argc, argv, "relocalization");
  ros::NodeHandle nh("~");
  ros::NodeHandlePtr nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

  Relocalization relocalization(nh_ptr);

  ros::MultiThreadedSpinner spinner(0);
  spinner.spin();
  return 0;
}