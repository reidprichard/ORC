set domainHeight 1
set domainLength 2

set fileName 2D-quad-between-mapped-layers_coarse
set verticalNodeCount 3
set horizontalNodeCount 5 

## Begins block of code that ctrl+z will undo so the entire script can be undone
ic_undo_group_begin

## Unload existing mesh
ic_unload_mesh

## Unload existing blocking
ic_hex_unload_blocking

## Unload existing geometry
ic_unload_tetin

## Delete empty parts (should all be empty)
ic_delete_empty_parts


ic_set_global geo_cad 0 toptol_userset
ic_set_global geo_cad 0.0 toler

ic_geo_new_family POINTS
ic_boco_set_part_color POINTS


#set scaleElementSize 0.004

## Create bounding box
ic_geo_cre_pnt POINTS pnt.1 {0 0 0}
ic_geo_cre_pnt POINTS pnt.2 "$domainLength 0 0"
ic_geo_cre_pnt POINTS pnt.3 "0 $domainHeight 0"
ic_geo_cre_pnt POINTS pnt.4 "$domainLength $domainHeight 0"

## Connect points with lines
ic_geo_new_family LINES
ic_geo_new_family INLET
ic_geo_new_family OUTLET
ic_geo_new_family TOP
ic_geo_new_family BOTTOM

## Bottom of bounding box
ic_geo_cre_line BOTTOM crv.00 pnt.1 pnt.2

## Left edge of bounding box
ic_geo_cre_line INLET crv.01 pnt.1 pnt.3

## Right edge of bounding box
ic_geo_cre_line OUTLET crv.02 pnt.2 pnt.4

## Top of bounding box
ic_geo_cre_line TOP crv.03 pnt.3 pnt.4

## Initialize 2D planar block
ic_geo_new_family FLUID
ic_hex_initialize_mesh 2d new_numbering new_blocking FLUID

## Note: 2D planar block vertices are initially numbered:
## 13 21
## 11 19
## Performing the first split creates vertices 33 and 34, increasing from bottom-to-top or left-to-right.
## Additional splits will assign the bottom-most or left-most new vertex a number 3 greater than the previous maximum number.
## For example, a second split parallel to the first split will create vertices 37 and 38,
## or a perpendicular second split will create vertices 37, 38 (where the two splits intersect), and 39.
## Each time a "selected" block is split, the index increases by 4 instead of 3.

## The first block is number 5, and the second (again, left-to-right and bottom-to-top) is number 10.
## Additional blocks follow the same rule: the first block created by a new split will be numbered the previous maximum plus 3,
## and additional blocks created by the same split will be sequentially greater by 1.
## For example, splitting a block horizontally will yield:
## 4 10
## Splitting this block again vertically will result in:
## 13 14
## 4  10
## Alternatively, performing the splits in the opposite order will give:
## 10 14
## 4  13
## *** Apparently, the above rule doesn't work for splitting "selected" blocks.

## Split block

## Associate bottom edge
ic_hex_set_edge_projection 11 19 0 1 crv.00

## Associate left edge
ic_hex_set_edge_projection 11 13 0 1 crv.01

## Associate right edge
ic_hex_set_edge_projection 19 21 0 1 crv.02

## Associate top edge
ic_hex_set_edge_projection 13 21 0 1 crv.03

## Set node count on left edges
ic_hex_set_mesh 11 13 n "$verticalNodeCount"

## Set node count on bottom edges
ic_hex_set_mesh 11 19 n "$horizontalNodeCount"


## Generates pre-mesh
ic_hex_create_mesh POINTS LINES INLET OUTLET TOP BOTTOM FLUID proj 2 dim_to_mesh 3 nproc 6

## Convert pre-mesh to unstructured mesh - don't really understand this part
ic_hex_write_file ./hex.uns POINTS LINES INLET OUTLET TOP BOTTOM FLUID proj 2 dim_to_mesh 2 no_boco
ic_uns_load ./hex.uns 3 0 {} 1
ic_uns_update_family_type visible {FLUID INLET BOTTOM OUTLET POINTS ORFN TOP LINES} {!NODE TRI_3 QUAD_4} update 0
ic_boco_solver
ic_boco_clear_icons

## Delete artifact LINES family
## Had to remove - this makes LINES an internal wall.
#ic_geo_delete_family LINES

## Chooses Fluent as solver
ic_boco_solver {ANSYS Fluent}

## Sets boundary conditions
## Pressure inlet
ic_boco_set INLET {{1 PRESI 0}}
## Wall
#ic_boco_set SCALE {{1 WALL 0}}
## Interior
#ic_boco_set FLUID {{1 INTER 0}}
## N/a
ic_boco_set ORFN {}
## ????
ic_boco_set POINTS { { 1  {color}  11194675  } }
## Pressure outlet
ic_boco_set OUTLET {{1 PRESO 0}}
## Wall
ic_boco_set BOTTOM {{1 WALL 0}}
## ???
ic_geo_new_family VORFN 0
## N/a
ic_boco_set VORFN {}
## Interior
ic_boco_set LINES {{1 INTER 0}}
## Wall
ic_boco_set TOP {{1 WALL 0}}

## Ends block of code that ctrl+z will undo so the entire script can be undone
ic_undo_group_end

## ************* Save mesh to .MSH file *******************



ic_chdir {C:/Users/Reid/orc/examples}
#ic_delete_empty_parts
#ic_delete_empty_parts

## Saves geometry: file only_visible [0] v4 [0] only_fams [""] only_ents [""] v10 [0] quiet [0] clear_undo [1]
#ic_save_tetin "$fileName.tin" 0 0 {} {} 0 0 1

## Checks for duplicate element and node numbers. Returns a list with two numbers - the number of dup elements and the number of dup nodes.
## Note that 0-numbered nodes and elements are not counted if skip_0 is 1 (the default).
# ic_uns_check_duplicate_numbers

## ic_save_unstruct file inc_subs [1] resnames [""] only_types [""] only_fams [""] only_subs [""] near_vols [0]
## Saves the current unstructured mesh to the given file. If the inc_subs argument is 1 (the default) then the current subsets are also saved.
## resnames is an optional list of result names to save with the domain (assuming they have been defined in the mesh).
## If only_subs is non-empty it is a list of names of maps that will be the only ones saved.
ic_save_unstruct "$fileName.uns" 1 {} {} {}

## Marks the mesh as modified. This is necessary for some reason; if not set, an error will be thrown.
ic_uns_set_modified 1

## Saves the current blocking
#ic_hex_save_blocking "$fileName.blk"

## Sets the default solver for an existing boco database. If no solver is given, then it returns the current solver setting.
#ic_boco_solver
ic_boco_solver {ANSYS Fluent}

## ???????????
ic_solution_set_solver {ANSYS Fluent} 1

## Saves the current boundary condition data to the given file name.
ic_boco_save "$fileName.fbc"

## Saves a new format bc (.atr) file.
#ic_boco_save_atr "$fileName.atr"

## ic_save_project_file file data [""] AppName [""]
## Saved project data to a file.
#ic_save_project_file {D:/OneDrive/Documents/Academics/Research/Sharkskin/Geometry, Meshing/scriptTest2.prj} {array\ set\ file_name\ \{ {    catia_dir .} {    parts_dir .} {    domain_loaded 0} {    cart_file_loaded 0} {    cart_file {}} {    domain_saved scriptTest2.uns} {    archive {}} {    med_replay {}} {    topology_dir .} {    ugparts_dir .} {    icons {{$env(ICEM_ACN)/lib/ai_env/icons} {$env(ICEM_ACN)/lib/va/EZCAD/icons} {$env(ICEM_ACN)/lib/icons} {$env(ICEM_ACN)/lib/va/CABIN/icons}}} {    tetin scriptTest2.tin} {    family_boco scriptTest2.fbc} {    iges_dir .} {    solver_params_loaded 0} {    attributes_loaded 0} {    project_lock {}} {    attributes scriptTest2.atr} {    domain scriptTest2.uns} {    domains_dir .} {    settings_loaded 0} {    settings scriptTest2.prj} {    blocking scriptTest2.blk} {    hexa_replay {}} {    transfer_dir .} {    mesh_dir .} {    family_topo {}} {    gemsparts_dir .} {    family_boco_loaded 0} {    tetin_loaded 0} {    project_dir .} {    topo_mulcad_out {}} {    solver_params {}} \} array\ set\ options\ \{ {    expert 1} {    remote_path {}} {    tree_disp_quad 2} {    tree_disp_pyra 0} {    evaluate_diagnostic 0} {    histo_show_default 1} {    select_toggle_corners 0} {    remove_all 0} {    keep_existing_file_names 0} {    record_journal 0} {    edit_wait 0} {    face_mode 0} {    select_mode all} {    med_save_emergency_tetin 1} {    user_name reide} {    diag_which all} {    uns_warn_if_display 500000} {    bubble_delay 1000} {    external_num 1} {    tree_disp_tri 2} {    apply_all 0} {    default_solver {ANSYS Fluent}} {    temporary_directory C:/Users/reide/AppData/Local/Temp} {    flood_select_angle 0} {    home_after_load 1} {    project_active 0} {    histo_color_by_quality_default 1} {    undo_logging 1} {    tree_disp_hexa 0} {    histo_solid_default 1} {    host_name DESKTOP-E384AG3} {    xhidden_full 1} {    replay_internal_editor 1} {    editor notepad} {    mouse_color orange} {    clear_undo 1} {    remote_acn {}} {    remote_sh csh} {    tree_disp_penta 0} {    n_processors 6} {    remote_host {}} {    save_to_new 0} {    quality_info Quality} {    tree_disp_node 0} {    med_save_emergency_mesh 1} {    redtext_color red} {    tree_disp_line 0} {    select_edge_mode 0} {    use_dlremote 0} {    max_mesh_map_size {}} {    show_tris 1} {    remote_user {}} {    enable_idle 0} {    auto_save_views 1} {    max_cad_map_size {}} {    display_origin 0} {    uns_warn_user_if_display 1000000} {    detail_info 0} {    win_java_help 0} {    show_factor 1} {    boundary_mode all} {    clean_up_tmp_files 1} {    auto_fix_uncovered_faces 1} {    med_save_emergency_blocking 1} {    max_binary_tetin 0} {    tree_disp_tetra 0} \} array\ set\ disp_options\ \{ {    uns_dualmesh 0} {    uns_warn_if_display 500000} {    uns_normals_colored 0} {    uns_icons 0} {    uns_locked_elements 0} {    uns_shrink_npos 0} {    uns_node_type None} {    uns_icons_normals_vol 0} {    uns_bcfield 0} {    backup Wire} {    uns_nodes 0} {    uns_only_edges 0} {    uns_surf_bounds 0} {    uns_wide_lines 0} {    uns_vol_bounds 0} {    uns_displ_orient Triad} {    uns_orientation 0} {    uns_directions 0} {    uns_thickness 0} {    uns_shell_diagnostic 0} {    uns_normals 0} {    uns_couplings 0} {    uns_periodicity 0} {    uns_single_surfaces 0} {    uns_midside_nodes 1} {    uns_shrink 100} {    uns_multiple_surfaces 0} {    uns_no_inner 0} {    uns_enums 0} {    uns_disp Wire} {    uns_bcfield_name {}} {    uns_color_by_quality 0} {    uns_changes 0} {    uns_cut_delay_count 1000} \} {set icon_size1 24} {set icon_size2 35} {set thickness_defined 0} {set solver_type 1} {set solver_setup -1} array\ set\ prism_values\ \{ {    n_triangle_smoothing_steps 5} {    min_smoothing_steps 6} {    first_layer_smoothing_steps 1} {    new_volume {}} {    height {}} {    prism_height_limit {}} {    interpolate_heights 0} {    n_tetra_smoothing_steps 10} {    do_checks {}} {    delete_standalone 1} {    ortho_weight 0.50} {    max_aspect_ratio {}} {    ratio_max {}} {    incremental_write 0} {    total_height {}} {    use_prism_v10 0} {    intermediate_write 1} {    delete_base_triangles {}} {    ratio_multiplier {}} {    verbosity_level 1} {    refine_prism_boundary 1} {    max_size_ratio {}} {    triangle_quality {}} {    max_prism_angle 180} {    tetra_smooth_limit 0.3} {    max_jump_factor 5} {    use_existing_quad_layers 0} {    layers 3} {    fillet 0.10} {    into_orphan 0} {    init_dir_from_prev {}} {    blayer_2d 0} {    do_not_allow_sticking {}} {    top_family {}} {    law exponential} {    min_smoothing_val 0.1} {    auto_reduction 0} {    stop_columns 1} {    stair_step 1} {    smoothing_steps 12} {    side_family {}} {    min_prism_quality 0.01} {    ratio 1.2} \} {set aie_current_flavor {}} array\ set\ vid_options\ \{ {    wb_import_mat_points 0} {    wb_NS_to_subset 0} {    wb_import_cad_att_pre {SDFEA;DDM}} {    wb_import_tritol 0.001} {    wb_import_mix_res -1} {    wb_import_save_pmdb {}} {    composite_tolerance 1.0} {    wb_import_save_partfile 0} {    wb_NS_to_entity_parts 0} {    wb_import_reference_key 0} {    replace 0} {    tdv_axes 1} {    vid_mode 0} {    auxiliary 0} {    wb_import_surface_bodies 1} {    show_name 0} {    wb_import_cad_att_trans 1} {    wb_import_solid_bodies 1} {    default_part GEOM} {    wb_import_mix_res_solid 0} {    new_srf_topo 1} {    DelPerFlag 0} {    wb_import_associativity_model_name {}} {    show_item_name 0} {    wb_import_work_points 0} {    wb_import_sel_proc 1} {    wb_NS_only 0} {    wb_import_scale_geo Default} {    wb_import_lcs 0} {    same_pnt_tol 1e-4} {    wb_import_transfer_file_scale 1.0} {    DelBlkPerFlag 0} {    wb_import_mesh 0} {    wb_import_mix_res_surface 0} {    wb_import_analysis_type 3} {    wb_import_geom 1} {    wb_import_refresh_pmdb 0} {    wb_import_load_pmdb {}} {    wb_import_mix_res_line 0} {    wb_import_delete_solids 0} {    inherit 1} {    wb_import_line_bodies 0} {    wb_import_en_sym_proc 1} {    wb_run_mesher tetra} {    wb_import_mix_res_point 0} {    wb_import_pluginname {}} {    wb_import_create_solids 0} {    wb_import_sel_pre {}} {    wb_import_cad_associativity 0} \} {set savedTreeVisibility {geomNode 1 geom_subsetNode 2 geomPointNode 0 geomCurveNode 2 meshNode 1 mesh_subsetNode 2 meshPointNode 0 meshLineNode 2 meshShellNode 2 meshTriNode 2 meshQuadNode 2 blockingNode 1 block_subsetNode 2 block_vertNode 0 block_edgeNode 2 block_faceNode 0 block_blockNode 0 block_meshNode 0 topoNode 2 topo-root 2 partNode 2 part-BOTTOM 2 part-FLUID 2 part-INLET 2 part-OUTLET 2 part-POINTS 2 part-SCALE 2 part-TOP 2 part-VORFN 2}} {set last_view {rot {0 0 0 1} scale {497.043108783 497.043108783 497.043108783} center {5.0000001192100001 0.5 0} pos {-437.732337011 172.697000179 0}}} array\ set\ cut_info\ \{ {    active 0} \} array\ set\ hex_option\ \{ {    default_bunching_ratio 2.0} {    floating_grid 0} {    project_to_topo 0} {    n_tetra_smoothing_steps 20} {    sketching_mode 0} {    trfDeg 1} {    wr_hexa7 0} {    smooth_ogrid 0} {    find_worst 1-3} {    hexa_verbose_mode 0} {    old_eparams 0} {    uns_face_mesh_method uniform_quad} {    multigrid_level 0} {    uns_face_mesh one_tri} {    check_blck 0} {    proj_limit 0} {    check_inv 0} {    project_bspline 0} {    hexa_update_mode 1} {    default_bunching_law BiGeometric} {    worse_criterion Quality} \} array\ set\ saved_views\ \{ {    views {}} \}} {ICEM CFD}

## ic_exec args
## A simplified version of this. ??????? Possibly referring to the argument ic_run_application_exec: Runs some external application, given the full path.
ic_exec {C:/Program Files/ANSYS Inc/v232/icemcfd/win64_amd/icemcfd/output-interfaces/fluent6} -dom "C:/Users/Reid/orc/examples/$fileName.uns" -b $fileName.fbc -dim2d -scale .001,.001,1.0 ./$fileName

## ****************** Evaluates mesh *********************

# ## ic_uns_num_couplings
# ## Counts the number of couplings.
# ic_uns_num_couplings

# #ic_undo_group_begin

# ## ic_uns_create_diagnostic_edgelist on
# ## Creates a temporary edge list for diagnostics.
# ic_uns_create_diagnostic_edgelist 1

# ## ic_uns_diagnostic args
# ## Runs diagnostics on the loaded unstructured mesh. The arguments come in keyword value pairs. The possible options are:
# ## See https://ansyshelp.ansys.com/account/secured?returnurl=/Views/Secured/corp/v201/en/icm_pguide/ipguide_meshdir_unstructmedfn.html for full list
# ## diag_type what : the type of diagnostic. This can be one or more of the following: ... all: all of the above
# ## fix_fam family : if fix is selected and new elements have to be created, put them in this family
# ## diag_verb : ?????
# ## fams famnames : check elements of the named families. This is just like the family argument except you can give multiple families.
# ##   If no family is given then all families are selected.
# ## busy_off : ?????
# ## quiet : ?????
# ic_uns_diagnostic subset all diag_type uncovered fix_fam FIX_UNCOVERED diag_verb {Uncovered faces} fams {} busy_off 1 quiet 1

# ## Must turn off diagnostic edgelist or something?
# ic_uns_create_diagnostic_edgelist 0
# #ic_undo_group_end

# ## ic_uns_min_metric crit [Quality] parts [""] types [""]
# ## Returns the min value (max value in case of Volume change) of a quality metric (and "" in case of an error).
# ## For example, set worst [ic_uns_min_metric "SPECIFIED_METRIC" "SPECIFIED_PARTS" "SPECIFIED_TYPES"]
# ## Default for SPECIFIED_METRIC is "Quality". Default for SPECIFIED_PARTS is all existing parts.
# ## Default for SPECIFIED_TYPES is all existing types (but skips NODE, LINE and BAR elements). Each of them can be empty.
# ## Must find minimum quality in entire mesh.
# ic_uns_min_metric Quality {} {}
