#pragma once

#include <chrono>
#include <iostream>

namespace adPredictAlgo {
struct Timer {
  typedef std::chrono::high_resolution_clock ClockT;
  typedef std::chrono::high_resolution_clock::time_point TimePointT;
  typedef std::chrono::high_resolution_clock::duration DurationT;
  typedef std::chrono::duration<double> SecondsT;

  TimePointT start;
  DurationT elapsed;
  Timer() { Reset(); }
  void Reset() {
    elapsed = DurationT::zero();
    Start();
  }
  void Start() { start = ClockT::now(); }
  void Stop() { elapsed += ClockT::now() - start; }
  double ElapsedSeconds() const { return SecondsT(elapsed).count(); }
/*  void PrintElapsed(std::string label) {
    printf("%s:\t %fs\n", label.c_str(), SecondsT(elapsed).count());
    Reset();
  }*/
};

}
