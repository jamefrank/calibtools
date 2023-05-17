#include <ros/ros.h>
#include <QApplication>
#include "form.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "main");
  ros::NodeHandle nh;
//  ROS_INFO("Hello world!");

  QApplication app(argc, argv);
  Form w;
  w.showMaximized();
  app.connect(&app, SIGNAL(lastWindowClosed()), &app, SLOT(quit()));
  int result = app.exec();

  return result;
}
