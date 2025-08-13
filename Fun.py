def fun(name):
    print(f"Hi, {name}")


fun("Ali")


def equation(x, y):
    return (x * 2 + y * 2 + 2 * x * y)


equation(3, 4)
equation(-3, 2)


def sq(x):
    return x * x


sq(5)


def cal(x, y):
    return x + y, x * y


cal(10, 15)


def rect_a_p(length, width):
    area = length * width
    perimeter = 2 * (length + width)
    return area, perimeter


rect_a_p(2, 6)

length = 5
width = 2
area, perimeter = rect_a_p(length, width)
print(f"area : {area}")
print(f"perimeter : {perimeter}")

import math


def area_circle(radius):
    return math.pi * radius * 2


area_circle(3)


def even(n):
    return n % 2 == 0


print(even(4))
print(even(7))


def m(a, b):
    return a if a > b else b


print(m(7, 12))


def fact(n):
    if n == 0:
        return 1
    return n * fact(n - 1)


print(fact(5))


def count_vowels(text):
    vowels = "aeiouAEIOU"
    return sum(1 for char in text if char in vowels)


print(count_vowels("Statistics and Data Science"))

print(count_vowels("Jahangirnagar University"))


def rev_string(st):
    return st[::-1]


print(rev_string("python"))


def count_words(sentence):
    return len(sentence.split())


print(count_words("More para in the Dept of SDS"))


def count_charecter(text, ch):
    return text.count(ch)


print(count_charecter("mango", "a"))
print(count_charecter("Jahangirnagar", "a"))


def sum_list(lst):
    return sum(lst)


print(sum_list([1, 2, 3, 4, 5]))


def largest(lst):
    return max(lst)


print(largest([4, 10, 20, 50, 34]))


def ave(nums):
    return sum(nums) / len(nums)


print(ave([10, 20, 30]))


def check_number(n):
    if n > 0:
        return "Positive"
    elif n < 0:
        return "Negaive"
    else:
        return "Zero"


print(check_number(-5))
print(check_number(5))
print(check_number(0))