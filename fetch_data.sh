

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=10cup9KnuvPbzk75sasc0giV3Nsn_tk_7' -O data/pollution_weather.csv

VAR="18CUS2nW1J5K9K-7Sx8gBreqkBpJBxu9G"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${VAR}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${VAR}" -o data/bus_station_boarding.zip
rm cookie

unzip data/bus_station_boarding.zip -d data/