#include<stdio.h>
/* crash course in C by MIT 
http://www.mit.edu/sipb/IAP/2004/c/CrashCourseC.html
 */
int main(void) {
	printf("review C programming..|_|\n");
	int c; /* Strict: variable must be declared before using */
	printf("rate the level of C programming you remembered: \n");
	scanf("%d", &c);
	printf("level = %d\n", c);

	/* all variable types */
	char c1 = 'a';
	float f1 = 930.15588e-2;
	double d1;
	printf("TESTING variable types\n");
	printf("c1 = %c, f1 = %.5f\n", c1, f1);


}
