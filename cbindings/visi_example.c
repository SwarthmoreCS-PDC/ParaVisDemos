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
 * (3) create worker pthreads, their main function should:
 *      init thread-specific data
 *      loop: do next computation step
 *            update color3 buf (call get_animation_buffer to get it)
 *            call draw_ready
 * (4) main thread call run_animation to start animation
 * (5) wait for worker threads to exit, clean-up
 *
 * (newhall, 2018)
 */
#include <pthreadGridVisi.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

#define DEFAULT_DIM    50
#define DEFAULT_INTERs  0
#define DEFAULT_TIDS    4
#define SLEEPYTIME   2000

/* (un)comment DEBUG to turn (on)off debugging */
//#define DEBUG  1
#ifdef DEBUG
#define PRINT(s) printf(s); fflush(stdout)
#define PRINT3(s,x,y,z) printf(s,x,y,z); fflush(stdout)
#else
#define PRINT(s) 
#define PRINT3(s,x,y,z) 
#endif

/* application-specific struct definition: needed for the visualizer */
struct appl_data {
   /* thread application data: */
   int *grid;  /* pointer to single shared grid */
   int iters;
   int curr_iter;  
   int rows;
   int cols;
   int mytid;
   int numtids;
   pthread_barrier_t *done;  /* pointer to single shared barrier */
   /* for using the visi library: */
   visi_handle handle; 
   color3 *image_buff;
};

static char visi_name[] = "pthreads!";

int init_state(struct appl_data *data, int r, int c, int tids, int iters);
void *thr_main(void *args);  
void update_grid(struct appl_data* data, int start_c, int stop_c); 

/**********************************************************/
int main(int argc, char *argv[]) {

  int cols, rows, iters, numtids, i;
  struct appl_data data;
  pthread_t *all_ptids;
  visi_handle myhandle;
  struct appl_data *info;  

  /* default values: */
  cols = DEFAULT_DIM;
  rows = DEFAULT_DIM;
  iters = DEFAULT_INTERs;
  numtids = DEFAULT_TIDS;

  /* (1) initalize application-specific state */
  //     (a) parse command line arguments
  if ((argc < 3) || (argc > 5)) {
    printf("usage: ./pthr_simple [rows cols] [num_tids] [num_iters]\n");
    exit(1);
  }
  if(argc >= 3) {
    rows = atoi(argv[1]);
    cols= atoi(argv[2]);
  } 
  if(argc >= 4) { numtids = atoi(argv[3]); }
  if(argc == 5) { iters = atoi(argv[4]); }
  if( ((rows%numtids) != 0) || ((cols%numtids) != 0) ) {
    printf("Error: num of threads %d must evenly divide dimensions %d:%d\n", 
        numtids, rows, cols);
    exit(1);
  }

  //    (b) alloc arrays for threads
  all_ptids = (pthread_t *)malloc(sizeof(pthread_t)*numtids);
  if(all_ptids == NULL) { 
    printf("ERROR malloc pthread array\n"); 
    exit(1); 
  }

  //    (c) create and initialize program data
  if(init_state(&data, rows, cols, numtids, iters) != 0) {
    printf("ERROR init world\n"); 
    exit(1);
  }
  printf("rows (height) = %d, cols (width) = %d thread %d iters %d\n", 
      rows, cols, numtids, iters);

  // (2) call init_pthread_animation
  myhandle = init_pthread_animation(numtids, rows, cols, 
      visi_name, iters);
  if(myhandle == NULL) {
    printf("ERROR init_pthread_animation\n"); 
    exit(1);
  }
  data.handle = myhandle;

  // (3) get the animation buffer 
  data.image_buff = get_animation_buffer(data.handle);
  if(data.image_buff == NULL) {
       printf("ERROR get_animation_buffer returned NULL\n"); 
       exit(1);
  }

  // (4) create worker threads passing each their thread-specific app data
  PRINT("about to spawn worker threads...\n");
  for(i=0; i < numtids; i++) {
    info = (struct appl_data *)malloc(sizeof(struct appl_data));
    if(!info) { printf("ERROR malloc info\n"); exit(1); }
    *info = data;  // init common fields
    info->mytid = i; // init thread specific fields
    if( pthread_create(&all_ptids[i], NULL, thr_main, (void *)info) ) {
      printf("pthread_created failed on thread %d...handle this better\n",i);
      exit(1);
    }
  }

  // (5) main thread calls run_animation to get it going
  run_animation(myhandle, iters);

  // (6) main thread waits for all threads to exit
  for(i=0; i < numtids; i++) {
     pthread_join(all_ptids[i], NULL);
  }

  // clean-up before exit
  free(all_ptids); all_ptids = NULL;
  free(data.grid);
  pthread_barrier_destroy(data.done);
  free(data.done);
}
/**********************************************************/
/* the thread main function:
 *   updates program state and 
 *   args: pointer to a struct appl_data for this thread
 *         (the thread is responsible for freeing the space before
 *         termination
 */
void *thr_main(void *args) {

  struct appl_data* data;
  int iters,i,j,index, c, r, chunksize, start_c, stop_c;

  /*  thread_specific state initalization:  */
  data = (struct appl_data *)args;
  iters = data->iters;
  c = data->cols;
  r = data->rows;
  chunksize = data->cols/data->numtids;
  start_c =  data->mytid*chunksize;
  stop_c = start_c + chunksize; 
  if(stop_c > c) { 
    stop_c = c; 
  }
  // init my part of the grid
  for(i=0; i < r; i++) {
    for(j=start_c; j < stop_c; j++) {
      index = i*c + j;
      data->grid[index] = j;
    }
  }
  // wait for all thread to init
  pthread_barrier_wait(data->done); 
  PRINT3("tid %d  start col = %d stop col = %d\n", data->mytid, 
      start_c, stop_c); 

  if(iters == 0 ) { 
    while(1) { /* run forever */
       update_grid(data, start_c, stop_c); 
       draw_ready(data->handle);
       usleep(SLEEPYTIME);
    }
  } else {
    for(i=0; i < iters; i++ ) {  /* run some number of iters */
       update_grid(data, start_c, stop_c); 
       draw_ready(data->handle);
       usleep(SLEEPYTIME);
    }
  }

  free(args);  args = NULL;
  return 0;
}
/**********************************************************/
/*
 * initialize application state that is common across all threads
 *  data: appl_data struct to initialize 
 *  r: number of rows
 *  c: number of cols
 *  i: number of iters
 *  returns: 0 on success 1 on error
 */
int init_state(struct appl_data *data, int r, int c, int t, int iters) {

  data->iters = iters;
  data->curr_iter = 0;
  data->rows = r;
  data->cols = c;
  data->numtids = t;
  data->grid = (int *)malloc(sizeof(int)*r*c);
  if(!data->grid) {
    printf("ERROR malloc\n"); 
    return 1;
  }

  data->done = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t));
  pthread_barrier_init(data->done, NULL, t);
  if(!data->grid) {
    printf("ERROR barrier malloc\n"); 
    return 1;
  }
  return 0;
}
/**********************************************************/
/*
 * performs one step of the application, called by individual tids
 *   data:  the application data
 *   start_c: the start column for this thread
 *   stop_c: the start column for this thread
 */
void update_grid(struct appl_data *data, int start_c, int stop_c){ 

  int i, j, r, c, index, buff_i, iter;
  color3 *buff;

  iter = data->curr_iter;   // just for readability
  buff = data->image_buff;  
  r = data->rows;
  c = data->cols;

  for(i=0; i < r; i++) {
    for(j=start_c; j < stop_c; j++) {
      index = i*c + j;
      // translate row index to y-coordinate value
      buff_i = (r - (i+1))*c + j;
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
