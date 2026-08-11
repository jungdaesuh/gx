#pragma once
#include "grids.h"
#include "parameters.h"
#include "geometry.h"
#include "reductions.h"
#include "device_funcs.h"
#include "grad_perp.h"
#include "nca.h"
#include "netcdf.h"
#include "netcdf_par.h"
#include <string>
#include "unistd.h"

using namespace std;


class NcDims;
class NcGrids;
class NcGeo;
class NcDiagnostics;

class NetCDF {
 public:
  NetCDF(Parameters* pars, Grids* grids, Geometry* geo = nullptr, string suffix = ".out.nc");
  ~NetCDF();

  int fileid;
  NcDims *nc_dims;
  NcGrids *nc_grids;
  NcGeo *nc_geo;
  NcDiagnostics *nc_diagnostics;
  void sync();
 private:
  void close_nc_file();
  Parameters *pars_;
  Grids *grids_;
  Geometry *geo_;
};

class NcDims {
 public:
  NcDims(Parameters *pars, Grids *grids, int fileid, bool append) {
    int retval;
    if (append) {
      if (retval = nc_inq_dimid (fileid, "ri",      &ri)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "x",       &x)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "y",       &y)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "theta",   &z)) ERR(retval);  
      if (retval = nc_inq_dimid (fileid, "kx",      &kx)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "ky",      &ky)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "kz",      &kz)) ERR(retval);  
      if (retval = nc_inq_dimid (fileid, "m",       &m)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "l",       &l)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "s",       &species)) ERR(retval);
      if (retval = nc_inq_dimid (fileid, "time",    &time)) ERR(retval);
    } else {
      if (retval = nc_def_dim (fileid, "ri",      2,                &ri)) ERR(retval);
      if (retval = nc_def_dim (fileid, "x",       pars->nx_in,     &x)) ERR(retval);
      if (retval = nc_def_dim (fileid, "y",       pars->ny_in,     &y)) ERR(retval);
      if (retval = nc_def_dim (fileid, "theta",   grids->Nz,       &z)) ERR(retval);  
      if (retval = nc_def_dim (fileid, "kx",      grids->Nakx,     &kx)) ERR(retval);
      if (retval = nc_def_dim (fileid, "ky",      grids->Naky,     &ky)) ERR(retval);
      if (retval = nc_def_dim (fileid, "kz",      grids->Nz,       &kz)) ERR(retval);  
      if (retval = nc_def_dim (fileid, "m",       pars->nm_in,     &m)) ERR(retval);
      if (retval = nc_def_dim (fileid, "l",       pars->nl_in,     &l)) ERR(retval);
      if (retval = nc_def_dim (fileid, "s",       pars->nspec_in,  &species)) ERR(retval);
      if (retval = nc_def_dim (fileid, "time",    NC_UNLIMITED,     &time)) ERR(retval);
    }
  };
  ~NcDims() {};

  int time, species, kx, ky, kz, x, y, z, l, m, ri;
};

class NcGrids {
 public:
  NcGrids(Grids* grids, NcDims* nc_dims, int fileid, bool append) {
    int retval;
    if (append) {
      if (retval = nc_inq_grp_ncid(fileid, "Grids", &grid_id)) ERR(retval);

      if (retval = nc_inq_varid(grid_id, "time", &time)) ERR(retval);
      if (retval = nc_inq_varid(grid_id, "kx", &kx)) ERR(retval);
      if (retval = nc_inq_varid(grid_id, "ky", &ky)) ERR(retval);
      if (retval = nc_inq_varid(grid_id, "kz", &kz)) ERR(retval);
      if (retval = nc_inq_varid(grid_id, "x", &x))  ERR(retval);  
      if (retval = nc_inq_varid(grid_id, "y", &y))  ERR(retval);  
      if (retval = nc_inq_varid(grid_id, "theta", &z))  ERR(retval);  

      if (retval = nc_var_par_access(grid_id, time, NC_COLLECTIVE)) ERR(retval);

      // read time_index
      if (retval = nc_inq_dimlen(fileid, nc_dims->time, &time_index)) ERR(retval);
    } else { 
      // define Grids group in ncdf
      if (retval = nc_def_grp(fileid, "Grids", &grid_id)) ERR(retval);

      if (retval = nc_def_var(grid_id, "time", NC_DOUBLE, 1, &nc_dims->time, &time)) ERR(retval);
      if (retval = nc_def_var(grid_id, "kx", NC_FLOAT, 1, &nc_dims->kx, &kx)) ERR(retval);
      if (retval = nc_def_var(grid_id, "ky", NC_FLOAT, 1, &nc_dims->ky, &ky)) ERR(retval);
      if (retval = nc_def_var(grid_id, "kz", NC_FLOAT, 1, &nc_dims->kz, &kz)) ERR(retval);
      if (retval = nc_def_var(grid_id, "x",  NC_FLOAT, 1, &nc_dims->x, &x))  ERR(retval);  
      if (retval = nc_def_var(grid_id, "y",  NC_FLOAT, 1, &nc_dims->y, &y))  ERR(retval);  
      if (retval = nc_def_var(grid_id, "theta",  NC_FLOAT, 1, &nc_dims->z, &z))  ERR(retval);  
 
      if (retval = nc_put_var(grid_id, kx, grids->kx_outh)) ERR(retval);
      if (retval = nc_put_var(grid_id, ky, grids->ky_h)) ERR(retval);
      if (retval = nc_put_var(grid_id, kz, grids->kz_outh)) ERR(retval);
      if (retval = nc_put_var(grid_id, x, grids->x_h)) ERR(retval);
      if (retval = nc_put_var(grid_id, y, grids->y_h)) ERR(retval);
      if (retval = nc_put_var(grid_id, z, grids->z_h)) ERR(retval);

      if (retval = nc_var_par_access(grid_id, time, NC_COLLECTIVE)) ERR(retval);
    }
  };
  ~NcGrids() {};
  void write_time(double time_val) {
    size_t count = 1;
    int retval;
    if (retval = nc_put_vara(grid_id, time, &time_index, &count, &time_val)) ERR(retval);
    time_index += 1;
  }

  int grid_id; // ncdf id for geo group
  // ncdf ids for grid variables
  int time, kx, ky, kz, x, y, z;

  size_t time_index = 0;
};

class NcGeo {
 public:
  NcGeo(Grids *grids, Geometry *geo, NcDims* nc_dims, int fileid) {
    int retval;
    // define Geometry group in ncdf
    if (retval = nc_def_grp(fileid, "Geometry", &geo_id)) ERR(retval);

    // define Geometry variables
    if (retval = nc_def_var (geo_id, "bmag",     NC_FLOAT, 1, &nc_dims->z, &bmag))     ERR(retval);
    if (retval = nc_def_var (geo_id, "bgrad",    NC_FLOAT, 1, &nc_dims->z, &bgrad))    ERR(retval);
    if (retval = nc_def_var (geo_id, "gbdrift",  NC_FLOAT, 1, &nc_dims->z, &gbdrift))  ERR(retval);
    if (retval = nc_def_var (geo_id, "gbdrift0", NC_FLOAT, 1, &nc_dims->z, &gbdrift0)) ERR(retval);
    if (retval = nc_def_var (geo_id, "cvdrift",  NC_FLOAT, 1, &nc_dims->z, &cvdrift))  ERR(retval);
    if (retval = nc_def_var (geo_id, "cvdrift0", NC_FLOAT, 1, &nc_dims->z, &cvdrift0)) ERR(retval);
    if (retval = nc_def_var (geo_id, "gds2",     NC_FLOAT, 1, &nc_dims->z, &gds2))     ERR(retval);
    if (retval = nc_def_var (geo_id, "gds21",    NC_FLOAT, 1, &nc_dims->z, &gds21))    ERR(retval);
    if (retval = nc_def_var (geo_id, "gds22",    NC_FLOAT, 1, &nc_dims->z, &gds22))    ERR(retval);
    if (retval = nc_def_var (geo_id, "grho",     NC_FLOAT, 1, &nc_dims->z, &grho))     ERR(retval);
    if (retval = nc_def_var (geo_id, "jacobian", NC_FLOAT, 1, &nc_dims->z, &jacobian)) ERR(retval);
    if (retval = nc_def_var (geo_id, "gradpar",  NC_FLOAT, 0, NULL, &gradpar))     ERR(retval);
    if (retval = nc_def_var (geo_id, "nperiod",  NC_INT, 0, NULL, &nperiod))     ERR(retval);
    if (retval = nc_def_var (geo_id, "q",  NC_FLOAT, 0, NULL, &q))     ERR(retval);
    if (retval = nc_def_var (geo_id, "shat",  NC_FLOAT, 0, NULL, &shat))     ERR(retval);
    if (retval = nc_def_var (geo_id, "shift",  NC_FLOAT, 0, NULL, &shift))     ERR(retval);
    if (retval = nc_def_var (geo_id, "rmaj",  NC_FLOAT, 0, NULL, &rmaj))     ERR(retval);
    if (retval = nc_def_var (geo_id, "aminor",  NC_FLOAT, 0, NULL, &aminor))     ERR(retval);
    if (retval = nc_def_var (geo_id, "kxfac",  NC_FLOAT, 0, NULL, &kxfac))     ERR(retval);
    if (retval = nc_def_var (geo_id, "drhodpsi",  NC_FLOAT, 0, NULL, &drhodpsi))     ERR(retval);
    if (retval = nc_def_var (geo_id, "theta_scale",  NC_FLOAT, 0, NULL, &theta_scale))     ERR(retval);
    if (retval = nc_def_var (geo_id, "nfp",  NC_INT, 0, NULL, &nfp))     ERR(retval);
    if (retval = nc_def_var (geo_id, "alpha",  NC_FLOAT, 0, NULL, &alpha))     ERR(retval);
    if (retval = nc_def_var (geo_id, "zeta_center",  NC_FLOAT, 0, NULL, &zeta_center))     ERR(retval);

    // write variables
    if (retval = nc_put_var(geo_id, bmag,     geo->bmag_h))     ERR(retval);
    if (retval = nc_put_var(geo_id, bgrad,    geo->bgrad_h))    ERR(retval);
    if (retval = nc_put_var(geo_id, gbdrift,  geo->gbdrift_h))  ERR(retval);
    if (retval = nc_put_var(geo_id, gbdrift0, geo->gbdrift0_h)) ERR(retval);
    if (retval = nc_put_var(geo_id, cvdrift,  geo->cvdrift_h))  ERR(retval);
    if (retval = nc_put_var(geo_id, cvdrift0, geo->cvdrift0_h)) ERR(retval);
    if (retval = nc_put_var(geo_id, gds2,     geo->gds2_h))     ERR(retval);
    if (retval = nc_put_var(geo_id, gds21,    geo->gds21_h))    ERR(retval);  
    if (retval = nc_put_var(geo_id, gds22,    geo->gds22_h))    ERR(retval);
    if (retval = nc_put_var(geo_id, grho,     geo->grho_h))     ERR(retval);
    if (retval = nc_put_var(geo_id, jacobian, geo->jacobian_h)) ERR(retval);
    if (retval = nc_put_var(geo_id, nperiod, &geo->nperiod))   ERR(retval);
    if (retval = nc_put_var(geo_id, gradpar, &geo->gradpar))   ERR(retval);
    if (retval = nc_put_var(geo_id, q, &geo->qsf))   ERR(retval);
    if (retval = nc_put_var(geo_id, shat, &geo->shat))   ERR(retval);
    if (retval = nc_put_var(geo_id, shift, &geo->shift))   ERR(retval);
    if (retval = nc_put_var(geo_id, rmaj, &geo->rmaj))   ERR(retval);
    if (retval = nc_put_var(geo_id, aminor, &geo->aminor))   ERR(retval);
    if (retval = nc_put_var(geo_id, kxfac, &geo->kxfac))   ERR(retval);
    if (retval = nc_put_var(geo_id, drhodpsi, &geo->drhodpsi))   ERR(retval);
    if (retval = nc_put_var(geo_id, theta_scale, &geo->theta_scale))   ERR(retval);
    if (retval = nc_put_var(geo_id, nfp, &geo->nfp))   ERR(retval);
    if (retval = nc_put_var(geo_id, alpha, &geo->alpha))   ERR(retval);
    if (retval = nc_put_var(geo_id, zeta_center, &geo->zeta_center))   ERR(retval);
  }
  ~NcGeo() {};
  int geo_id; // ncdf id for geo group
  // ncdf ids for geo variables
  int bmag, bgrad, gbdrift, gbdrift0, cvdrift, cvdrift0;
  int gds2, gds21, gds22, grho, jacobian, gradpar;
  int q, shat, shift, kxfac, rmaj, aminor, drhodpsi, theta_scale, nperiod, nfp;
  int alpha, zeta_center;
};

class NcDiagnostics {
 public:
  NcDiagnostics(int fileid, bool append) {
    int retval;
    if (append) {
      if (retval = nc_inq_grp_ncid(fileid, "Diagnostics", &diagnostics_id)) ERR(retval);
    } else {
      // create ncdf group id for Diagnostics
      if (retval = nc_def_grp(fileid, "Diagnostics", &diagnostics_id)) ERR(retval);
    }
  };
  ~NcDiagnostics() {};
  int diagnostics_id; // ncdf id for diagnostics group
};
