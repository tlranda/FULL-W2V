#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <cstring>

using namespace std;

#define MAX_EXP 6

#ifdef HALF_PRECISION
  typedef half real;
  #define PREC "%.16hf"
#else
  #ifdef DOUBLE_PRECISION
    typedef double real;
    #define PREC "%.64lf"
  #else
    // #define SINGLE_PRECISION
    typedef float real;
    #define PREC "%.32f"
  #endif
#endif

int main(int argc, char **argv) {
  if(argc < 2) {
      printf("USAGE: %s <Size of EXPTABLE> [-no-cuda]\n", argv[0]);
      printf("\t-no-cuda: Omit output of cuda type classes in definition\n");
      printf("It's recommended to redirect stdout to a file to capture the output\n");
      printf("Defaults to single precision. Recompile with -D [HALF_PRECISION, DOUBLE_PRECISION] to change precision\n");
      exit(1);
  }
  int size = atoi(argv[1]);
  bool cuda = true;
  if(argc > 2 && !strcmp(argv[2], "-no-cuda")) {
     cuda = false;
  }

  char fmt_str[10], eol_fmt_str[10], fin_fmt_str[10];
  sprintf(fmt_str, "%s, ", PREC);
  sprintf(eol_fmt_str, "%s,\n", PREC);
  sprintf(fin_fmt_str, "%s", PREC);
  if (cuda)
      printf("__constant__ __restrict__ ");
  else
      printf("const ");
  printf("real expTable[%d] = {", size);
  for(int i = 0; i < size; i++) {
      real val = exp((i / (real) size * 2 - 1) * MAX_EXP);
      val = val / (val + 1);
      if (i == 0) printf(fmt_str, 0.0);
      else if(i + 1 == size) printf(fin_fmt_str, 1.0);
      else if(i % 3 == 2) printf(eol_fmt_str, val);
      else printf(fmt_str, val);
  }
  printf("};\n");


  if (cuda)
      printf("__constant__ __restrict__ ");
  else
      printf("const ");
  printf("real nexpTable[%d] = {", size);
  for(int i = size-1; i >= 0; i--) {
      real val = exp((i / (real) size * 2 - 1) * MAX_EXP);
      val = val / (val + 1);
      if (i == size-1) printf(fmt_str, 1.0);
      else if(i == 0) printf(fin_fmt_str, 0.0);
      else if(i % 3 == 2) printf(eol_fmt_str, val);
      else printf(fmt_str, val);
  }
  printf("};");

  return 0;
}
