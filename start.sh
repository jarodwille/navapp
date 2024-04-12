# start up the server
export LD_LIBRARY_PATH="/home/jj/thesis/libtorch/lib:/home/jj/thesis/valhalla/build/src/model"
valhalla_service valhalla.json 1
# curl it directly if you like:
# curl http://localhost:8002/route --data '{"locations":[{"lat":47.365109,"lon":8.546824,"type":"break","city":"Zürich","state":"Altstadt"},{"lat":47.108878,"lon":8.394801,"type":"break","city":"6037 Root","state":"Untere Waldstrasse"}],"costing":"auto","directions_options":{"units":"miles"}}' | jq '.'

#HAVE FUN!