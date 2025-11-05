#include "STDesc.h"
#include "slict/FeatureCloud.h"
#include "utility.h"
#include <optional>

#include <sensor_msgs/Imu.h>

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
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/io/pcd_io.h>

#include "PreintBase.h"
#include <factor/PreintFactor.h>

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"
#define RESET "\033[0m"

struct TimedCloud
{
  CloudXYZIPtr cloud;
  ros::Time stamp;
};

static inline double to_ms(const std::chrono::steady_clock::time_point &t0,
                           const std::chrono::steady_clock::time_point &t1)
{
  using namespace std::chrono;
  return duration_cast<duration<double, std::milli>>(t1 - t0).count();
}

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
  // ros::Subscriber imuSub;
  ros::Publisher relocPub;
  ros::Publisher alignedCloudPub;

  // Queues + mutex
  std::deque<TimedCloud> cloud_queue;

  // std::deque<sensor_msgs::ImuConstPtr> imu_queue;
  std::mutex cloud_mutex;
  // std::mutex imu_mutex;

  const std::deque<sensor_msgs::ImuConstPtr> *imu_queue_shared = nullptr;
  std::mutex *imu_mutex_shared = nullptr;
  double last_imu_used_time = -1.0;

  // Global prior map + KD‐tree
  CloudXYZIPtr priorMap;
  KdFLANNPtr kdTreeMap;

  // Thread control
  std::thread processThread;
  std::atomic<bool> running{true};

  // Timing
  double startTime = 0.0;
  double last_cloud_time = -1.0;

  // Pose state
  mytf initPose;
  mytf prevPose;
  int liteloam_id;

  // IMU pre‐integration
  std::unique_ptr<PreintBase> imuPreint;

  // ===== ICP tuning (coarse -> fine) =====
  int iterCoarse = 10;
  int iterFine = 15;
  double transEps = 1e-6;
  double fitEps = 1e-5;
  double ransacThresh = 0.05;
  bool use_recip_corr = false;

  double maxcorrCoarse = 10.0;
  double maxcorrFine = 3.0;

  float leafCoarse = 0.20f;
  float leafFine = 0.08f;

  double icpThresh = 2.0;

  bool icpConverged = false;
  bool posePublished = false;

  bool firstCloud = true;
  bool icpOnly = true;
  bool oneShot = true;

  // imu
  Eigen::Vector3d v_prev = Eigen::Vector3d::Zero();              // velocity at prev frame
  Eigen::Vector3d v_curr = Eigen::Vector3d::Zero();              // velocity at current frame
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();                  // accel bias
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();                  // gyro bias
  double acc_n = 0.6, acc_w = 0.08, gyr_n = 0.05, gyr_w = 0.003; // default; load from ROS if muốn
  Eigen::Vector3d gvec = Eigen::Vector3d(0, 0, 9.81);

public:
  // Constructor
  LITELOAM(const CloudXYZIPtr &priorMap,
           const KdFLANNPtr &kdt,
           const mytf &initPose,
           int liteloam_id,
           const ros::NodeHandlePtr &nh_ptr,
           const std::optional<TimedCloud> &seed_cloud,
           const std::deque<sensor_msgs::ImuConstPtr> *imu_q_shared,
           std::mutex *imu_mtx_shared)
      : nh_ptr(nh_ptr),
        priorMap(priorMap),
        kdTreeMap(kdt),
        initPose(initPose),
        prevPose(initPose),
        liteloam_id(liteloam_id)
  {
    ros::NodeHandle pnh("~"); // private handle

    // ===== Read ICP tuning from ROS params (with sensible defaults) =====
    pnh.param("iterCoarse", iterCoarse, iterCoarse);
    pnh.param("iterFine", iterFine, iterFine);
    pnh.param("maxcorrCoarse", maxcorrCoarse, maxcorrCoarse);
    pnh.param("maxcorrFine", maxcorrFine, maxcorrFine);
    pnh.param("leafCoarse", leafCoarse, leafCoarse);
    pnh.param("leafFine", leafFine, leafFine);
    pnh.param("icpThresh", icpThresh, icpThresh);

    pnh.param("icpOnly", icpOnly, true);
    pnh.param("oneShot", oneShot, true);

    kdTreeMap->setInputCloud(priorMap);

    if (seed_cloud.has_value())
    {
      std::lock_guard<std::mutex> lk(cloud_mutex);
      cloud_queue.push_back(*seed_cloud); // first frame is the matched one
    }

    imu_queue_shared = imu_q_shared;
    imu_mutex_shared = imu_mtx_shared;

    lidarCloudSub = nh_ptr->subscribe("/lastcloud", 100, &LITELOAM::PCHandler, this);
    // imuSub = nh_ptr->subscribe("/vn100/imu", 500, &LITELOAM::IMUHandler, this);
    relocPub = nh_ptr->advertise<geometry_msgs::PoseStamped>("/liteloam_pose", 100);
    alignedCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/liteloam_aligned_cloud", 1);

    startTime = ros::Time::now().toSec();
    running = true;
    processThread = std::thread(&LITELOAM::processBuffer, this);

    ROS_INFO_STREAM("[LITELOAM " << liteloam_id << "] Constructed");
  }

  ~LITELOAM()
  {
    shutdown();
    ROS_WARN("[LITELOAM %d] destructed.", liteloam_id);
  }

  void stop() { running = false; }

  double timeSinceStart() const
  {
    return ros::Time::now().toSec() - startTime;
  }

  void shutdown()
  {
    running = false;

    if (processThread.joinable())
      processThread.join();

    lidarCloudSub.shutdown();
    // imuSub.shutdown();
    alignedCloudPub.shutdown();
    relocPub.shutdown();
  }

  bool loamConverged() const { return icpConverged; }
  bool published() const { return posePublished; }
  bool isRunning() const { return running; }
  int getID() const { return liteloam_id; }

private:
  // void IMUHandler(const sensor_msgs::ImuConstPtr &msg)
  // {
  //   std::lock_guard<std::mutex> lk(imu_mutex);
  //   imu_queue.push_back(msg);
  // }

  void PCHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    CloudXYZIPtr cloud(new CloudXYZI());
    pcl::fromROSMsg(*msg, *cloud);

    std::lock_guard<std::mutex> lk(cloud_mutex);
    cloud_queue.push_back({cloud, msg->header.stamp});
  }

  void processBuffer()
  {
    while (running && ros::ok())
    {
      bool printFirst = firstCloud;

      // 1) wait for cloud
      if (cloud_queue.empty())
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      // 2) pop cloud
      TimedCloud item;
      {
        std::lock_guard<std::mutex> lk(cloud_mutex);
        item = cloud_queue.front();
        cloud_queue.pop_front();
      }

      auto t_frame_start = std::chrono::steady_clock::now();

      CloudXYZIPtr raw = item.cloud;
      ros::Time cloudTime = item.stamp;
      double tNow = cloudTime.toSec();
      double tPrev = (last_cloud_time < 0.0 ? tNow : last_cloud_time);
      last_cloud_time = tNow;
      double dt = tNow - tPrev;
      (void)dt; // unused but kept for potential future use

      // --- 4) Two-level downsampling (coarse + fine) ---
      // Coarse keeps structure, reduces outliers; Fine preserves details for precise alignment.
      CloudXYZIPtr srcCoarse(new CloudXYZI()), srcFine(new CloudXYZI());
      {
        pcl::VoxelGrid<PointXYZI> vg;
        // Coarse downsample from raw
        vg.setInputCloud(raw);
        vg.setLeafSize(leafCoarse, leafCoarse, leafCoarse);
        vg.filter(*srcCoarse);

        // Fine downsample directly from raw to preserve details
        vg.setInputCloud(raw);
        vg.setLeafSize(leafFine, leafFine, leafFine);
        vg.filter(*srcFine);
      }

      // 5) collect IMU in (tStart, tNow]
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

      // printf("[IMU] samples: %zu, dt: %.3f s\n", imuBuf.size(), tNow - tPrev);

      // 6) IMU pre-integration (minimal)
      if (!imuBuf.empty())
      {
        Eigen::Vector3d a0(imuBuf.front()->linear_acceleration.x,
                           imuBuf.front()->linear_acceleration.y,
                           imuBuf.front()->linear_acceleration.z);
        Eigen::Vector3d g0(imuBuf.front()->angular_velocity.x,
                           imuBuf.front()->angular_velocity.y,
                           imuBuf.front()->angular_velocity.z);

        if (!imuPreint)
        {
          imuPreint.reset(new PreintBase(
              a0, g0, // first sample
              ba, bg, // current biases
              true,
              acc_n, acc_w, // noises
              gyr_n, gyr_w,
              gvec, // gravity
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
      else
      {
        imuPreint.reset(nullptr);
      }

      // // --- 6.5) Predict initial pose using IMU pre-integration ---
      mytf predPose = prevPose;
      if (imuPreint)
      {
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
        predPose.pos = prevPose.pos + (prevPose.rot * dp);
        predPose.rot = (prevPose.rot * dq).normalized(); // optional
      }

      // --- 7) Coarse -> Fine ICP against priorMap ---
      // Strategy:
      //   1) Coarse pass: broader search (bigger max_corr, fewer constraints) to find a reasonable basin.
      //   2) Fine pass: tighter search (smaller max_corr, more iterations) for accuracy.
      // We also test a small yaw sweep around the IMU-seeded orientation to avoid local minima.

      Eigen::Matrix4f bestTrans = Eigen::Matrix4f::Identity();
      double bestFitness = std::numeric_limits<double>::infinity();

      // Base pose from IMU seed
      Eigen::Quaternionf q0 = predPose.rot.cast<float>();
      q0.normalize();
      Eigen::Matrix3f R0 = q0.toRotationMatrix();
      Eigen::Vector3f T0 = predPose.pos.cast<float>();

      // Yaw offsets: larger search for first cloud, narrower later
      std::vector<double> yawOffsets;
      if (firstCloud)
      {
        for (double d = -180.0; d <= 180.0; d += 30.0)
          yawOffsets.push_back(d);
      }
      else
      {
        double d = 0.0;
        yawOffsets.push_back(d);
      }

      // Helper to run PCL ICP with consistent settings
      auto run_icp = [&](const CloudXYZIPtr &src,
                         double max_corr,
                         int max_iter,
                         const Eigen::Matrix4f &init,
                         Eigen::Matrix4f &T_out,
                         double &fitness) -> bool
      {
        // pcl::GeneralizedIterativeClosestPoint<PointXYZI, PointXYZI> icp;
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

      Eigen::Matrix4f coarse_best_T = Eigen::Matrix4f::Identity();
      double coarse_best_fit = std::numeric_limits<double>::infinity();

      // 7a) Coarse pass: try multiple yaw initializations
      for (double d : yawOffsets)
      {
        double yawRad = d * M_PI / 180.0;

        if (yawRad > M_PI)
          yawRad -= 2.0 * M_PI;
        else if (yawRad < -M_PI)
          yawRad += 2.0 * M_PI;

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

      // If coarse pass fails completely, skip this frame
      if (!std::isfinite(coarse_best_fit))
      {
        printf(KYEL
               "[LITELOAM %d] Coarse ICP failed. \n" RESET,
               liteloam_id);
        // raise flag
        icpConverged = false;
        running = false;
        continue;
      }

      // 7b) Fine pass: tighten distances and iterate more for precision
      Eigen::Matrix4f T_fine;
      double fit_fine;
      if (!run_icp(srcFine, maxcorrFine, iterFine, coarse_best_T, T_fine, fit_fine))
      {
        printf(KYEL
               "[LITELOAM %d] Fine ICP failed. \n" RESET,
               liteloam_id);
        // raise flag
        icpConverged = false;
        running = false;
        continue;
      }

      bestTrans = T_fine;
      bestFitness = fit_fine;

      // Use stricter fitness thresholds to accept only accurate alignments
      double fitness_threshold = firstCloud
                                     ? icpThresh
                                     : icpThresh;
      if (bestFitness >= fitness_threshold)
      {

        printf(KYEL
               "[LITELOAM %d] ICP fitness: %.4f IMUBuf: %zu ICP NOT CONVERGED\n"
               "Pinit: %7.2f. %7.2f. %7.2f. YPR: %4.0f. %4.0f. %4.0f.\n" RESET,
               liteloam_id, bestFitness, imuBuf.size(),
               predPose.pos.x(), predPose.pos.y(), predPose.pos.z(),
               predPose.yaw(), predPose.pitch(), predPose.roll());
        icpConverged = false;
        running = false;
        continue;
      }

      // // 8) Build pose & refine with IMU factor
      Eigen::Matrix4d bestTransD = bestTrans.cast<double>();
      mytf finalPose(bestTransD);
      if (finalPose.rot.w() < 0.0)
        finalPose.rot.coeffs() *= -1.0;

      if (imuPreint)
      {
        // param blocks
        double pi[7] = {prevPose.pos.x(), prevPose.pos.y(), prevPose.pos.z(),
                        prevPose.rot.x(), prevPose.rot.y(), prevPose.rot.z(), prevPose.rot.w()};
        double pj[7] = {finalPose.pos.x(), finalPose.pos.y(), finalPose.pos.z(),
                        finalPose.rot.x(), finalPose.rot.y(), finalPose.rot.z(), finalPose.rot.w()};
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

        for (int k = 0; k < 3; ++k)
        {
          problem.SetParameterLowerBound(baj, k, -0.5); // ba
          problem.SetParameterUpperBound(baj, k, 0.5);
          problem.SetParameterLowerBound(baj, 3 + k, -0.1); // bg
          problem.SetParameterUpperBound(baj, 3 + k, 0.1);
        }

        // Residual IMU
        ceres::CostFunction *f_preint = new PreintFactor(imuPreint.get());
        problem.AddResidualBlock(f_preint, nullptr, pi, vi, bai, pj, vj, baj);

        ceres::Solver::Options opts;
        opts.max_num_iterations = 20;
        opts.max_solver_time_in_seconds = 0.02;
        opts.linear_solver_type = ceres::DENSE_QR;
        opts.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary sum;
        ceres::Solve(opts, &problem, &sum);

        finalPose.pos.x() = pj[0];
        finalPose.pos.y() = pj[1];
        finalPose.pos.z() = pj[2];
        finalPose.rot = Eigen::Quaterniond(pj[6], pj[3], pj[4], pj[5]).normalized();

        v_curr = Eigen::Vector3d(vj[0], vj[1], vj[2]);

        ba = Eigen::Vector3d(baj[0], baj[1], baj[2]);
        bg = Eigen::Vector3d(baj[3], baj[4], baj[5]);
      }

      // 9) Publish
      if (icpOnly || !firstCloud)
      {
        auto t_before_publish = std::chrono::steady_clock::now();
        const double proc_ms = to_ms(t_frame_start, t_before_publish);
        const ros::Duration proc_dur(proc_ms / 1000.0);

        ros::Time stamp_out = cloudTime + proc_dur;

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

        printf(KGRN
               "[LITELOAM %d] ICP fitness: %.4f. FC: %s. Stamp: %.3f IMUBuf: %zu\n"
               "Pinit: %7.2f. %7.2f. %7.2f. YPR: %4.0f. %4.0f. %4.0f.\n"
               "Plast: %7.2f. %7.2f. %7.2f. YPR: %4.0f. %4.0f. %4.0f.\n" RESET,
               liteloam_id, bestFitness,
               printFirst ? "True" : "False",
               ps.header.stamp.toSec(),
               imuBuf.size(),
               prevPose.pos.x(), prevPose.pos.y(), prevPose.pos.z(),
               prevPose.yaw(), prevPose.pitch(), prevPose.roll(),
               finalPose.pos.x(), finalPose.pos.y(), finalPose.pos.z(),
               finalPose.yaw(), finalPose.pitch(), finalPose.roll());

        sensor_msgs::PointCloud2 outMsg;
        pcl::PointCloud<PointXYZI> aligned;
        pcl::transformPointCloud(
            *srcFine, aligned, finalPose.tfMat().cast<float>());
        pcl::toROSMsg(aligned, outMsg);
        outMsg.header.stamp = cloudTime;
        outMsg.header.frame_id = "map";
        alignedCloudPub.publish(outMsg);

        // raise flag
        posePublished = true;
        if (oneShot)
          running = false;
      }

      // 10) update
      firstCloud = false;
      icpConverged = true;
      imuPreint.reset(nullptr);
      v_prev = v_curr;
      prevPose = finalPose;
    }

    printf("[LITELOAM %d] exiting. \n", liteloam_id);
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

  std::atomic<bool> active_{true};
  int liteloamNum = 5;
  bool oneShot = true;

  ros::Subscriber imuSub;

  std::deque<sensor_msgs::ImuConstPtr> imu_queue_shared;
  std::mutex imu_mutex_shared;

  int imu_max_keep = 12000;
  double imu_max_age = 30.0;

public:
  ~Relocalization()
  {
    if (checkLoamThread.joinable())
      checkLoamThread.join();
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
      bool should_quit = false;

      {
        std::lock_guard<std::mutex> lg(loam_mtx);
        for (auto &loam : loamInstances)
        {
          if (!loam)
            continue;

          if (loam->timeSinceStart() > 10.0 && !loam->loamConverged() && loam->isRunning())
          {
            printf("[Relocalization] LITELOAM %d exceeded 10s. Stop it.\n", loam->getID());
            loam->stop();
          }
          else if (loam->published() && oneShot)
          {
            should_quit = true;
          }
        }
      }

      if (should_quit)
      {
        std::vector<std::shared_ptr<LITELOAM>> locals;
        {
          std::lock_guard<std::mutex> lg(loam_mtx);
          locals.swap(loamInstances);
        }
        for (auto &lm : locals)
          if (lm)
            lm->shutdown();

        lidarCloudSub.shutdown();
        ulocSub.shutdown();
        imuSub.shutdown();

        active_.store(false);
        ROS_INFO("[Relocalization] All LiteLOAM instances stopped. Shutting down node...");

        ros::requestShutdown();
        return;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  void Initialize()
  {
    // ULOC pose subscriber
    ulocSub =
        nh_ptr->subscribe("/uwb_pose", 100, &Relocalization::ULOCCallback, this);

    lidarCloudSub = nh_ptr->subscribe("/lastcloud", 100, &Relocalization::PCHandler, this);

    nh_ptr->param("imu_max_keep", imu_max_keep, imu_max_keep);
    nh_ptr->param("imu_max_age", imu_max_age, imu_max_age);

    imuSub = nh_ptr->subscribe("/vn100/imu", 1000, &Relocalization::IMUHandler, this);

    nh_ptr->param("liteloamNum", liteloamNum, liteloamNum);
    ROS_INFO("[RELOC] Number of liteloam: %d", liteloamNum);

    // loadPriorMap
    string prior_map_dir = "";
    nh_ptr->param("/priormap_dir", prior_map_dir, string(""));
    ROS_INFO("[RELOC] priormap_dir: %s", prior_map_dir.c_str());

    this->priorMap.reset(new pcl::PointCloud<pcl::PointXYZI>());
    this->kdTreeMap.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());

    std::string pcd_file = prior_map_dir + "/priormap.pcd";

    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *priorMap) != 0)
    {
      ROS_ERROR("[RELOC] Failed to load PCD: %s", pcd_file.c_str());
      return;
    }
    else
    {
      ROS_INFO("[RELOC] Prebuilt pcd map found, loading...");
    }

    ROS_INFO("[RELOC] Prior Map (%zu points).", priorMap->size());

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

    ROS_INFO("[RELOC] Downsampled Prior Map (%zu points).", priorMap->size());

    // Update kdTree with the new downsampled map
    this->kdTreeMap->setInputCloud(this->priorMap);
    priorMapReady = true;

    ROS_INFO("[RELOC] Prior Map Load Completed \n");

    nh_ptr->param("/oneShot", oneShot, true);
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

    std::lock_guard<std::mutex> lk(imu_mutex_shared);
    imu_queue_shared.push_back(m);

    while (imu_queue_shared.size() > imu_max_keep)
      imu_queue_shared.pop_front();

    const double t_now = imu_queue_shared.back()->header.stamp.toSec();
    while (!imu_queue_shared.empty() &&
           (t_now - imu_queue_shared.front()->header.stamp.toSec()) > imu_max_age)
      imu_queue_shared.pop_front();
  }

  // Match UWB pose to the LiDAR cloud by NEAREST timestamp (simplified)
  void ULOCCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
  {
    if (!priorMapReady)
    {
      ROS_WARN("[RELOC] Prior map not yet loaded; skipping UWB callback");
      return;
    }

    // Buffer the incoming UWB pose
    {
      std::lock_guard<std::mutex> lock(uloc_mutex);
      uloc_buffer.push_back(msg);
      if (uloc_buffer.size() > 10000)
        uloc_buffer.pop_front();
    }

    TimedCloud matched_cloud;
    {
      std::lock_guard<std::mutex> lock(cloud_mutex);
      if (cloud_queue.empty())
        return;
      matched_cloud = cloud_queue.back(); // copy
    }

    // Get the timestamp of the most recent LiDAR cloud
    ros::Time latest_cloud_time;
    {
      latest_cloud_time = matched_cloud.stamp;
    }

    // Pick the UWB pose with MIN absolute time difference
    geometry_msgs::PoseStamped::ConstPtr best_pose = nullptr;
    double best_diff = std::numeric_limits<double>::infinity();
    {
      std::lock_guard<std::mutex> lock(uloc_mutex);
      for (auto &pose_msg : uloc_buffer)
      {
        double diff = std::abs((pose_msg->header.stamp - latest_cloud_time).toSec());
        if (diff < best_diff)
        {
          best_diff = diff;
          best_pose = pose_msg;
        }
      }
    }

    if (!best_pose)
    {
      ROS_WARN("[RELOC] Could not find a matching ULOC for cloud at %f",
               latest_cloud_time.toSec());
      return;
    }

    {
      std::lock_guard<std::mutex> lock(uloc_mutex);

      auto it = std::find_if(uloc_buffer.begin(), uloc_buffer.end(),
                             [&](const geometry_msgs::PoseStamped::ConstPtr &p)
                             {
                               return p.get() == best_pose.get();
                             });
      if (it != uloc_buffer.end())
        uloc_buffer.erase(it);
    }

    // Wrap the matched pose into your transform type
    mytf start_pose(*best_pose);

    // Spawn or restart a LITELOAM instance with that start pose
    {
      std::lock_guard<std::mutex> lg(loam_mtx);

      if (loamInstances.size() < liteloamNum)
      {
        int newID = static_cast<int>(loamInstances.size());
        auto inst = std::make_shared<LITELOAM>(
            priorMap, kdTreeMap, start_pose, newID, nh_ptr, std::make_optional(matched_cloud), &imu_queue_shared, &imu_mutex_shared);
        loamInstances.push_back(inst);
        printf("[RELOC] Created LITELOAM %d (matched ULOC at %f). \n",
               newID, best_pose->header.stamp.toSec());
      }
      else
      {
        for (auto &loam : loamInstances)
        {
          if (!loam->isRunning())
          {
            int id = loam->getID();
            loam = std::make_shared<LITELOAM>(
                priorMap, kdTreeMap, start_pose, id, nh_ptr, std::make_optional(matched_cloud), &imu_queue_shared, &imu_mutex_shared);
            printf("[RELOC] Restarted LITELOAM %d (matched ULOC at %f). \n",
                   id, best_pose->header.stamp.toSec());
            break;
          }
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
