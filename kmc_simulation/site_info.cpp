#include "site_info.h"

SiteInfo::SiteInfo() {}

SiteInfo::SiteInfo(int central_type, int bottom_neighbor, int lateral_f_count, int lateral_n_count, int top_neighbor):
  central_type(central_type),
  bottom_neighbor(bottom_neighbor),
  lateral_f_count(lateral_f_count),
  lateral_n_count(lateral_n_count),
  top_neighbor(top_neighbor)
{
  total_counts = get_total_counts();
}


SiteInfo::~SiteInfo() {}


void SiteInfo::update_counts() {
  axial_n_count = (bottom_neighbor==NANOPARTICLE) + (top_neighbor==NANOPARTICLE);
  total_n_count = lateral_n_count + (bottom_neighbor==NANOPARTICLE) + (top_neighbor==NANOPARTICLE);
  total_f_count = lateral_f_count + (bottom_neighbor==FLUID) + (top_neighbor==FLUID);
  total_s_count = bottom_neighbor==SURFACE;
}


std::string SiteInfo::to_string() const {
  return std::to_string(central_type) + std::to_string(bottom_neighbor) + std::to_string(lateral_f_count) + std::to_string(lateral_n_count) + std::to_string(top_neighbor);
}


std::string SiteInfo::get_total_counts() {
  update_counts();
  return std::to_string(central_type) + std::to_string(lateral_n_count) + std::to_string(axial_n_count) + std::to_string(total_f_count) + std::to_string(total_s_count);
}
