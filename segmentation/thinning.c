#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

/* 
 * The array we're working on looks like this:
 * P9  P2  P3
 * P8  P1  P4
 * P7  P6  P5
 * 
 * this corresponds with the following array indices:
 * 0   1   2
 * 3   4   5
 * 6   7   8
 */

const unsigned int
    P1 = 4, // (1, 1)
    P2 = 1, // (1, 0)
    P3 = 2, // (2, 0) 
    P4 = 5, // (2, 1)
    P5 = 8, // (2, 2)
    P6 = 7, // (1, 2)
    P7 = 6, // (0, 2)
    P8 = 3, // (0, 1)
    P9 = 0; // (0, 0)

// Indices of a flattened 3x3 array starting at the top middle index and moving
// around it clockwise, not including the center pixel
const unsigned int ORDER[] = { P2, P3, P4, P5, P6, P7, P8, P9 };

// Indices of a flattened 3x3 array starting at the top middle index and moving
// around it clockwise, including the first index twice (once at the beginning
// and once at the end)
const unsigned int LOOP[] = { P2, P3, P4, P5, P6, P7, P8, P9, P2 };

void print_window(double window[]) {
    printf("%1.0f %1.0f %1.0f\n", window[0], window[1], window[2]);
    printf("%1.0f %1.0f %1.0f\n", window[3], window[4], window[5]);
    printf("%1.0f %1.0f %1.0f\n", window[6], window[7], window[8]);
}

void print_order(double window[]) {
    for (int i = 0; i < 8; i++) {
        double item = window[ORDER[i]];
        printf("P%d: %1.0f; ", i + 2, item);
    }
    printf("\n");
}

// Calculates the number of 0 -> 1 transitions in the sequence
static unsigned int A(
    double *buffer
) {
    int transitions = 0;
    for (int i = 0; i < 9; i++) {
        // (mis)using the fact that booleans in C are represented by 0 and 1
        // ~p & q is true (=1) only when p = 0 & q = 1
        int cur = buffer[LOOP[i]];
        int nxt = buffer[LOOP[i + 1]];
        transitions += (!cur && nxt);
    }

    return transitions;
}

// Calculates the number of nonzero neighbours
static unsigned int B(
    double *buffer
) {
    int sum = 0;
    for (int i = 0; i < 8; i++) {
        int cur = buffer[ORDER[i]];
        sum += cur;
    }
    return sum;
}

// Checks if our 3x3 window only contains 0s and 1s
static bool IsBinary(
    double *buffer
) {
    bool isBinary = true;

    for (int i = 0; i < 9; i++) {
        if (buffer[i] != 0 && buffer[i] != 1) {
            isBinary = false;
        }
    }

    return isBinary;
}

int StepOne(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data
) {
    // This only works with a 3x3 filter with only 0s and 1s
    if (filter_size != 9 || !IsBinary(buffer)) {
        // 0 is failure in CPython
        return 0;
    }

    // The pixel is black and has eight neighbours
    bool c0 = buffer[P1] == 1;
    // 2 <= B (P1) <= 6 
    unsigned int b = B(buffer);
    bool c1 = 2 <= b && b <= 6;
    // A(P1) = 1
    bool c2 = A(buffer) == 1;
    // At least one of P2 and P4 and P6 is white
    bool c3 = buffer[P2] * buffer[P4] * buffer[P6] == 0;
    // At least one of P4 and P6 and P8 is white
    bool c4 = buffer[P4] * buffer[P6] * buffer[P8] == 0;

    // return 0 if all conditions are true
    *return_value = (double) !(c0 && c1 && c2 && c3 && c4);

    // 1 is success in CPython
    return 1;
}

int StepTwo(
    double *buffer,
    intptr_t filter_size,
    double *return_value,
    void *user_data
) {
    // This only works with a 3x3 filter with only 0s and 1s
    if (filter_size != 9 || !IsBinary(buffer)) {
        // 0 is failure in CPython
        return 0;
    }

    // The pixel is 1
    bool c0 = buffer[P1] == 1;
    // 2 <= B (P1) <= 6 
    unsigned int b = B(buffer);
    bool c1 = 2 <= b && b <= 6;
    // A(P1) = 1
    bool c2 = A(buffer) == 1;
    // At least one of P2 and P4 and P8 is white
    bool c3= buffer[P2] * buffer[P4] * buffer[P8] == 0;
    // At least one of P2 and P6 and P8 is white
    bool c4 = buffer[P2] * buffer[P6] * buffer[P8] == 0;

    // return 0 if all conditions are true
    *return_value = (double) !(c0 && c1 && c2 && c3 && c4);

    // 1 is success in CPython
    return 1;
}

// For testing
int main(int argc, char const *argv[])
{
    double test1[] = { 1, 1, 1, 
                       0, 1, 1,
                       0, 1, 1};

    double test1_a = 0;
    double test1_b = 0;
    double test1_result = 999;
    double test2_result = 999;
    printf("IsBinary: %d\n", IsBinary(test1));
    printf("A: %u\n", A(test1));
    printf("B: %u\n", B(test1));

    print_window(test1);
    print_order(test1);
    
    StepOne(test1, 9, &test1_result, NULL);
    printf("\nStep 1: %1.0f\n", test1_result);
    double test1_2[9];
    memcpy(test1_2, test1, 9*sizeof(double));
    test1_2[4] = test1_result;
    print_window(test1_2);
    print_order(test1_2);
    
    StepTwo(test1, 9, &test2_result, NULL);
    printf("\nStep 2: %1.0f\n", test2_result);
    double test2_2[9];
    memcpy(test2_2, test1, 9*sizeof(double));
    test2_2[4] = test2_result;
    print_window(test2_2);
    print_order(test2_2);

    return 0;
}