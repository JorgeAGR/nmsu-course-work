#include <stdio.h>

int main() {
	int a[3][3] = {
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
		      };
	
	int * pa = a;
	
	for (int i = 0; i < 9; i++){
		printf("%d\n", *pa);
		pa++;
	}

	for (int i = 0; i < 3; i++){
		printf("%ls\n", a[i + 6]);
	}
	return 0;
}
