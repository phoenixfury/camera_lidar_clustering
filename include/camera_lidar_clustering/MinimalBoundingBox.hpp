// Copyright 16.04.2021. Alexander Kormanovsky

#ifndef MinimalBoundingBox_MinimalBoundingBox
#define MinimalBoundingBox_MinimalBoundingBox

#include <vector>

namespace MinimalBoundingBoxNS
{

/**
 * Port from
 * https://github.com/cansik/LongLiveTheSquare
 * https://github.com/cansik/LongLiveTheSquare/blob/master/U4LongLiveTheSquare/Geometry/GeoAlgos.cs
 * https://github.com/cansik/LongLiveTheSquare/blob/8aab5069763c0e1d8c451195d8855cb713aee48b
 * /U4LongLiveTheSquare/MinimalBoundingBox.cs
 */
class MinimalBoundingBox
{
public:
  struct Point
  {
    double x = 0;
    double y = 0;

    Point() {}

    Point(double x, double y)
    {
      this->x = x;
      this->y = y;
    }

    inline Point operator+(const Point & other) const
    {
      return {this->x + other.x, this->y + other.y};
    }

    inline Point operator-(const Point & other) const
    {
      return {this->x - other.x, this->y - other.y};
    }
  };

  struct Segment
  {
    Point a, b;

    Segment(const Point & a, const Point & b)
    {
      this->a = a;
      this->b = b;
    }
  };

  struct Rect
  {
    Point location;
    Point size;

    Rect() {}

    Rect(const Point & a, const Point & c)
    {
      location = a;
      size = c - a;
    }

    bool is_empty() { return location.x == 0 && location.y == 0 && size.x == 0 && size.y == 0; }

    double get_area() { return size.x * size.y; }

    std::vector<Point> get_points() const
    {
      return {
        {location.x, location.y},
        {location.x + size.x, location.y},
        {location.x + size.x, location.y + size.y},
        {location.x, location.y + size.y}};
    }
  };

  struct BoundingBox
  {
    std::vector<Point> boundingPoints;
    std::vector<Point> hullPoints;
    Point center;
    // smaller box side
    double width;
    // larger box side
    double height;
    // angle between smaller box side and X axis in radians,
    // positive value means box orientation from bottom right to top left,
    // negative value means opposite
    double widthAngle;
    // angle between larger box side and X axis in radians
    // positive value means box orientation from bottom left to top right,
    // negative value means opposite
    double heightAngle;
    // weather the box is aligned to axes (widthAngle is 0 and height angle is 90)
    bool isAligned;
  };

public:
  static BoundingBox calculate(const std::vector<Point> & points, double alignmentTolerance = 0.0);

private:
  static double cross(const Point & o, const Point & a, const Point & b);

  static bool is_double_equal(double v1, double v2);

  static std::vector<Point> monotone_chain_convex_hull(const std::vector<Point> & points);

  static double angle_to_X_axis(const Segment & s);

  static Point rotate_to_X_axis(const Point & p, double angle);
};

}  // namespace MinimalBoundingBoxNS

#endif  // MinimalBoundingBox_MinimalBoundingBox
