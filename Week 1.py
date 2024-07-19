#Create lower triangular, upper triangular and pyramid containing the "*" character.


def lowertriangular(n):
    for i in range(1, n + 1):
        print('* ' * i)

def uppertriangular(n):
    for i in range(n, 0, -1):
        print('* ' * i)

def pyramid(n):
    for i in range(1, n + 1):
        print(' ' * (n - i) + '* ' * i)

n = 5

print("Lower Triangular Pattern:")
lowertriangular(n)
print("\nUpper Triangular Pattern:")
uppertriangular(n)
print("\nPyramid Pattern:")
pyramid(n)
