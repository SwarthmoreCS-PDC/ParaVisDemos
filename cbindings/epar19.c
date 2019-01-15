/*
 * Example of using C wrapper around using pthreads with the
 * the cgridvisi library (part of the qtogl lib):
 *
 * To run: (note number of threads must evenly divide rows and cols)
 *   ./a.out
 *   ./a.out 100 100           # w/optional command line args for dimensions
 *   ./a.out 500 500 10        # rows cols num_tids
 *   ./a.out 500 500 10 300    # rows cols num_tidsnumber_of_iteratons
 *
 * Need to define:
 * (0) a struct containing all program specific data
 *     including a color3 * for the visi buffer to color
 *
 * The main control flow:
 * (1) initialize all program specific data
 * (2) call init_pthread_animation
 * (2) call get_buff_pthread_animation
 * (3) create worker pthreads, their main function should:
 *      init thread-specific data
 *      loop: do next computation step
 *            update color3 buf
 *            call draw_pthread_animation
 * (4) main thread call run_pthread_animation to
 * (5) wait for worker threads to exit, clean-up
 */
#include <pthreadGridVisi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define DEFAULT_DIM 100
#define DEFAULT_INTERS 150
#define DEFAULT_TIDS 4
#define SLEEPYTIME 2000


// application-specific struct definition: needed for the visualizer
struct appl_data {
  /* thread application data: */
  int *grid; /* pointer to single shared grid */
  int iters;
  int curr_iter;
  int rows;
  int cols;
  int mytid;
  int numtids;
  pthread_barrier_t *done;
  /* fields for visi: */
  visi_handle handle; /* pass to visi library functions */
  color3 *image_buff;         /* pointer to single shared image buffer */
};



int local_init_state(struct appl_data *data, int r, int c, int tids, int iters);
void *local_thread_main(void *args);
void local_update_grid(struct appl_data *data, int start_c, int stop_c);


/**********************************************************/


int main(int argc, char *argv[]) {

  int cols, rows, iters, numtids, i;
  visi_handle myhandle;
  static char visi_name[] = "pthreads!";
  struct appl_data *thread_info;
  pthread_t *all_ptids;

  /* init common thread info */
  cols = rows = iters = 100;
  numtids = 4;
  all_ptids = malloc(sizeof(pthread_t) * numtids);
  thread_info = malloc(sizeof(struct appl_data) * numtids);
  local_init_state(&thread_info[0], rows, cols, numtids, iters);

  /* main thread gets handle and image buffer from library  */
  myhandle = init_pthread_animation(numtids, rows, cols, visi_name);
  thread_info[0].handle = myhandle;
  thread_info[0].image_buff = get_animation_buffer(myhandle);

  /* create threads and pass a copy of handle to each
     thread through its thread_info field */
  for (i = 0; i < numtids; i++) {
    thread_info[i] = thread_info[0];    // init common fields
    thread_info[i].mytid = i; // init thread specific fields
    pthread_create(&all_ptids[i], NULL, local_thread_main,
       (void *)(&thread_info[i]));
  }

  /* main thread triggers animation on handle */
  run_animation(myhandle, iters);

  /* wait for exit, cleanup*/
  for (i = 0; i < numtids; i++) {
    pthread_join(all_ptids[i], NULL);
  }
  free(all_ptids);
  free(thread_info);
}
/**********************************************************/
void *local_thread_main(void *args) {

  struct appl_data *data;
  int iters, i, j, index, c, r, chunksize, start_c, stop_c;

  /*  thread_specific state initalization:  */
  data = (struct appl_data *)args;
  iters = data->iters;
  c = data->cols;
  r = data->rows;
  chunksize = data->cols / data->numtids;
  start_c = data->mytid * chunksize;
  stop_c = start_c + chunksize;
  if (stop_c > c) {
    stop_c = c;
  }
  // init my part of the grid
  for (i = 0; i < r; i++) {
    for (j = start_c; j < stop_c; j++) {
      index = i * c + j;
      data->grid[index] = j;
    }
  }
  // wait for all thread to init
  pthread_barrier_wait(data->done);


  if (iters == 0) {
    while (1) { /* run forever */
      local_update_grid(data, start_c, stop_c);
      /* draw ready is a library function that blocks until
         all other threads are also ready */
      draw_ready(data->handle);
      usleep(SLEEPYTIME);
    }
  } else {
    for (i = 0; i < iters; i++) { /* run some number of iters */
      local_update_grid(data, start_c, stop_c);
      draw_ready(data->handle);
      usleep(SLEEPYTIME);
    }
  }

  return 0;
}
/**********************************************************/
/*
 * returns 0 on success 1 on error
 */
int local_init_state(struct appl_data *data, int r, int c, int t, int iters) {

  // TODO: add error checking of params
  data->iters = iters;
  data->curr_iter = 0;
  data->rows = r;
  data->cols = c;
  data->numtids = t;
  data->grid = (int *)malloc(sizeof(int) * r * c);

  data->done = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t));
  pthread_barrier_init(data->done, NULL, t);

  return 0;
}
/**********************************************************/
void local_update_grid(struct appl_data *data, int start_c, int stop_c) {

  int i, j, r, c, index, buff_i, iter;
  color3 *buff;

  iter = data->curr_iter; // just for readability
  buff = data->image_buff;
  r = data->rows;
  c = data->cols;

  for (i = 0; i < r; i++) {
    for (j = start_c; j < stop_c; j++) {
      index = i * c + j;
      // translate row index to y-coordinate value
      buff_i = (r - (i + 1)) * c + j;
      // update animation buffer
      buff[buff_i].r = (data->grid[index]) % 256;
      buff[buff_i].g = (data->grid[index] + start_c) % 256;
      buff[buff_i].b = 200;
      // change grid for next round
      data->grid[index] = (data->grid[index] + 10) % 256;
    }
  }
  data->curr_iter = iter + 1;
  // force threads to wait until all are done before each starts next
  // (this synch is not necessary here because threads don't share any
  // computation state, but is just FYI)
  pthread_barrier_wait(data->done);
}
