// timer.h
#ifndef TIMER_H
#define TIMER_H

typedef struct {
    struct timespec start;
    struct timespec end;
} Timer;

void start_timer(Timer *t);
void stop_timer(Timer *t);
double get_elapsed_sec(const Timer *t);

#endif
