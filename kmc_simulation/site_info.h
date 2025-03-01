#ifndef SI_
#define SI_

#include <string>

enum CellType {
  SURFACE = 0,
  FLUID = 1,
  NANOPARTICLE = 2
};
  
class SiteInfo {
  public:
    SiteInfo();
    SiteInfo(int, int, int, int, int);
    ~SiteInfo();
    
    std::string to_string() const;
    std::string get_total_counts();

    int central_type;
    int bottom_neighbor;
    int lateral_f_count;
    int lateral_n_count;
    int top_neighbor;
    int total_n_count = 0;
    int total_f_count = 0;
    int total_s_count = 0;
    int axial_n_count = 0;
    std::string site_id;
    std::string total_counts;
    
private:
  void update_counts();
};
  
#endif