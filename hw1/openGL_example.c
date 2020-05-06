#include <GL/freeglut.h> 
#include <stdlib.h> 
#include <stdio.h>
#include <unistd.h>

#define screenX 1366
#define screenY 768

float r[screenX][screenY],g[screenX][screenY],b[screenX][screenY];

void idle()
{                      

	for(int i=0;i<screenX;i++)
	{
		for(int j=0;j<screenY;j++)
		{
			r[i][j]=(float)((rand() % 9))/8;
			g[i][j]=(float)((rand() % 9))/8;
			b[i][j]=(float)((rand() % 9))/8;
		}
	
	}	
	usleep(100000); //sleep 0.1 second
	glutPostRedisplay();

}

void magic_dots(void)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, screenX, 0.0, screenY);

	

	for(int i=0;i<screenX;i++)
	{
		for(int j=0;j<screenY;j++)
		{
			glColor3f(r[i][j],g[i][j],b[i][j]); 
			glBegin(GL_POINTS);
			glVertex2i (i,j);
			glEnd();
		}
	
	}		

	

	glFlush();	
}


int main(int argc,char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_SINGLE);
	glutInitWindowSize(screenX, screenY);
	glutCreateWindow("Randomly generated points");
	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT);
	glutDisplayFunc(magic_dots);
	glutIdleFunc(idle);
	glutMainLoop();
	
	return 0;
}