# download some data and make tiles out of it
# NOTE: you can feed multiple extracts into pbfgraphbuilder
# wget http://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf
# get the config and setup
# mkdir -p valhalla_tiles
# valhalla_build_config --mjolnir-tile-dir ${PWD}/valhalla_tiles --mjolnir-tile-extract ${PWD}/valhalla_tiles.tar --mjolnir-timezone ${PWD}/valhalla_tiles/timezones.sqlite --mjolnir-admin ${PWD}/valhalla_tiles/admins.sqlite > valhalla.json
# # build timezones.sqlite to support time-dependent routing
# valhalla_build_timezones > valhalla_tiles/timezones.sqlite
# # build admins.sqlite to support admin-related properties such as access restrictions, driving side, ISO codes etc
# valhalla_build_admins -c valhalla.json new-york-latest.osm.pbf
# # build routing tiles
# valhalla_build_tiles -c valhalla.json new-york-latest.osm.pbf
# # tar it up for running the server
# # either run this to build a tile index for faster graph loading times
# valhalla_build_extract -c valhalla.json -v