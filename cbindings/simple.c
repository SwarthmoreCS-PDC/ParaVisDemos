/*
 * Example of using the cgridvisi library:
 *
 * To run:
 *   ./simple
 *   ./simple 100 100        # w/optional command line args for dimensions
 *   ./simple 500 500 300    # rows cols number_of_iteratons
 *
 * Need to define:
 * (0) a struct containing all program specific data
 * (1) a main loop function with the prototype:
 *     void change_world(color3 *buff, void* appl_data);
 *     (a) does some (the next step) of computation
 *     (b) updates the visualization buff to visualize next step
 *
 * The main functionl control flow:
 * (1) initialize all program specific data in application struct
 * (2) call init_and_run_animation passing in application struct and
 *      name of function main loop function
 *
 */
#include <cgridvisi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// application-specific struct definition: needed for the visualizer
struct appl_data {
  int *grid;
  int iters;
  int rows;
  int cols;
};

// application-specific main function: definition needed for the visualizer
void change_world(color3 *buff, void *data);

static char visi_name[] = "SimpleVisi";

int init_world(struct appl_data *data, int r, int c);

/**********************************************************/
int main(int argc, char *argv[]) {

  struct appl_data data;
  int cols = 50;
  int rows = 50;
  int iter = 0;

  // (1) initalize application-specific state

  // parse command line arguments
  // optional args for rows and cols dimensions and number of iterations
  //     ./simple 100 100 500
  if (argc == 3 || argc == 4) {
    rows = atoi(argv[1]);
    cols = atoi(argv[2]);
  } else if (argc > 4) {
    printf("usage: ./simple [num_rows num_cols] [num_iters]\n");
    exit(1);
  }
  if (argc == 4) {
    iter = atoi(argv[3]);
  }
  printf("rows (height) = %d, cols (width) = %d iters %d\n", rows, cols, iter);

  // create and initialize program data
  if (init_world(&data, rows, cols)) {
    printf("ERROR init world\n");
    exit(1);
  }

  // (2) call init and run animation
  init_and_run_animation(rows, cols, &data, change_world, visi_name, iter);
}

/**********************************************************/
/*
 * returns 0 on success 1 on error
 */
int init_world(struct appl_data *data, int r, int c) {

  int i, j, index;

  data->iters = 0;
  data->rows = r;
  data->cols = c;
  data->grid = (int *)malloc(sizeof(int) * r * c);
  if (!data->grid) {
    printf("ERROR malloc\n");
    return 1;
  }
  for (i = 0; i < r; i++) {
    for (j = 0; j < c; j++) {
      index = i * c + j;
      data->grid[index] = 0;
      if (i == j) {
        data->grid[index] = 1;
      }
    }
  }
  return 0;
}
/**********************************************************/
void change_world(color3 *buff, void *adata) {

  struct appl_data *data;
  int i, j, r, c, iters, index, buff_i;

  data = (struct appl_data *)adata;
  iters = data->iters;
  r = data->rows;
  c = data->cols;

  for (i = 0; i < r; i++) {
    for (j = 0; j < c; j++) {
      index = i * c + j;
      // translate row index to y-coordinate value
      buff_i = (r - (i + 1)) * c + j;
      // update animation buffer
      if (data->grid[index] == 1) {
        // darker blue at top to lighter blue
        buff[buff_i].r = (50 + 2 * i) % 256;
        buff[buff_i].g = (50 + 2 * i) % 256;
        buff[buff_i].b = 200;
      } else {
        buff[buff_i].r = 0;
        buff[buff_i].g = 0;
        buff[buff_i].b = 0;
      }
      data->grid[index] = 0;
      // change grid for next round
      if (((i + iters) % r) <= j) {
        data->grid[buff_i] = 1;
      }
    }
  }
  data->iters = iters + 1;
}
