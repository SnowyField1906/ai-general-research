def group_by(iterable, property):
    grouped = {}

    for item in iterable:
        try:
            grouped[item[property]].append(item)
        except:
            grouped[item[property]] = [item]

    return grouped

def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def intersection(iterable1, iterable2):
    if len(iterable1) <= len(iterable2):
        little, big = iterable1, iterable2
    else:
        little, big = iterable2, iterable1

    result = set()
    big = sorted(big) # O(nlogn)
    
    for e in little: # O(mlogn)
        if binary_search(big, e) != -1:
            result.add(e)
    
    return result

if __name__ == "__main__":
    data = [
        { "name": "nhi", "age": "17"},
        { "name": "kiet", "age": "18"},
        { "name": "long", "age": "17"},
        { "name": "thuan", "age": "16"}
    ]

    grouped = group_by(data, "age")
    print("grouped:", grouped)

    age1 = list(grouped.keys()) # ["17", "18", "16"]
    age2 = ["15", "16", "17"]
    intersected = intersection(age1, age2)
    print("intersected:", intersected)