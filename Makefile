
CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors


symnmf: symnmf.c symnmf.h 
	$(CC) $(CFLAGS) -o symnmf symnmf.c -lm

clean:
	rm -f symnmf *.o *.so

	