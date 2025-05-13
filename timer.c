// timer.c
#include <time.h>
#include "timer.h"

void start_timer(Timer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

void stop_timer(Timer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->end);
}

double get_elapsed_sec(const Timer *t) {
    long seconds = t->end.tv_sec - t->start.tv_sec;
    long nanoseconds = t->end.tv_nsec - t->start.tv_nsec;
    
    if (nanoseconds < 0) {
        seconds--;
        nanoseconds += 1000000000;
    }
    return (double)seconds + (double)nanoseconds * 1e-9;
}
