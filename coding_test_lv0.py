## 배열뒤집기
def solution(num_list):
    num_list.reverse()
    #num_list[::-1]
    return num_list

## 특정 문자 제거하기
def solution(my_string, letter):
    answer=my_string.replace(letter,"")
    return answer 

## 모음 제거
def solution(my_string):
    a = ['a','e','i','o','u']
    for i in a:
        my_string= my_string.replace(i,'')
    return my_string

## 아이스 아메리카노
def solution(money):
    answer = [money//5500,money%5500]
    return answer
