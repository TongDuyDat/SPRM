
try:
    S = input()
    K = int(input()) % 26  
    res = ""
    for c in S:
        if 'a' <= c <= 'z':
            res += chr((ord(c) - ord('a') + K) % 26 + ord('a'))
        elif 'A' <= c <= 'Z':
            res += chr((ord(c) - ord('A') + K) % 26 + ord('A'))
        else:
            res += c 

    print(res, end="")

except Exception as e:
    print("", end="")