nums = [3,3,-1,0,1,2,-3,-3]

def get_sum_0(nums):

    nums = sorted(nums)

    i = 0
    j = len(nums) - 1

    res = []

    while( i < j):
        if nums[i] + nums[j] < 0:
            i += 1
        elif nums[i] + nums[j] > 0:
            j -= 1
        else:
            if nums[i] ==  nums[i+1]:
                i += 1
                continue
            elif nums[j] == nums[j-1]:
                j -= 1
                continue
            res.append([nums[i],nums[j]])
            i += 1
    
    return res

print(get_sum_0(nums=nums))
