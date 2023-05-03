Usage: ./render <size> <input_file> <renderer> <sim cycles> <pixel_size> <output_file>

./render 500 ./inputs/random_500.txt cpu 20 4 no-output
./render 1000 ./inputs/random_1000.txt cpu 20 4 no-output
./render 2500 ./inputs/random_2500.txt cpu 20 4 no-output
./render 5000 ./inputs/random_5000.txt cpu 20 4 no-output

./render 500 ./inputs/random_500.txt openmp 20 4 no-output
./render 1000 ./inputs/random_1000.txt openmp 20 4 no-output
./render 2500 ./inputs/random_2500.txt openmp 20 4 no-output
./render 5000 ./inputs/random_5000.txt openmp 20 4 no-output

./render 500  ./inputs/random_500.txt  cpu 20 2 no-output
./render 1000 ./inputs/random_1000.txt cpu 20 2 no-output
./render 2500 ./inputs/random_2500.txt cpu 20 2 no-output
./render 5000 ./inputs/random_5000.txt cpu 20 2 no-output

./render 500  ./inputs/random_500.txt  gpu 20 2 no-output
./render 1000 ./inputs/random_1000.txt gpu 20 2 no-output
./render 2500 ./inputs/random_2500.txt gpu 20 2 no-output
./render 5000 ./inputs/random_5000.txt gpu 20 2 no-output
