def compare_and_swap(arr1,arr2,arr3,arr4):
    arr5=[0 for i in range(len(arr1))]
    for i in range(len(arr1)):
        arr5[i]=(arr1[i]>arr2[i])*arr3[i]+(arr1[i]<=arr2[i])*arr4[i]
    return arr5

def rotate(arr, n):
    # Ensure the rotation is within the length of the array
    n = n % len(arr)
    return arr[n:] + arr[:n]


def sort_simu(arr):
    n = len(arr)
    print("n must be power of 2: ",n)
    k = 2 # the block size at each stage
    while k<=n:
        j = k//2 # initial step_size
        while j > 0:
            print("arr: ",arr)
            mask1=[0 for i in range(n)]
            mask2=[0 for i in range(n)]
            mask3=[0 for i in range(n)]
            mask4=[0 for i in range(n)]
            # mask making
            for i in range(n):
                l = i^j
                if l>i:
                    if(i&k == 0):
                        mask1[i]=1
                        mask2[l]=1
                    else:
                        mask3[i]=1
                        mask4[l]=1
            print(mask1)
            arr1=[0 for i in range(n)]
            arr2=[0 for i in range(n)]
            arr3=[0 for i in range(n)]
            arr4=[0 for i in range(n)]

            # print("k: ",k,"j: ",j,"l: ",l)

            # element wise mult for
            for i in range(n):
                arr1[i]=arr[i]*mask1[i]
                arr2[i]=arr[i]*mask2[i]
                arr3[i]=arr[i]*mask3[i]
                arr4[i]=arr[i]*mask4[i]

            arr5 = rotate(arr1,-j) # right shift
            # print("arr5: ",arr5)
            arr6 = rotate(arr2,j) # left shift
            # print("arr6: ",arr6)
            arr5_2 = rotate(arr3,-j) # right shift
            arr6_2 = rotate(arr4,j)
            arr7=[0 for i in range(n)]
            for i in range(n):
                arr7[i] = arr6[i]+arr5[i]+arr6_2[i]+arr5_2[i] # used for comparison
            # print("arr7: ",arr7)
            arr8 = arr #used for comparison
            # print("arr8: ",arr8)
            arr9 = [0 for i in range(n)]
            for i in range(n):
                arr9[i]=arr5[i]+arr1[i]+arr6_2[i]+arr4[i]
            # print("arr9: ",arr9)
            arr10 = [0 for i in range(n)]
            for i in range(n):
                arr10[i] = arr6[i]+arr2[i]+arr5_2[i]+arr3[i]
            # print("arr10: ",arr10)
            arr = compare_and_swap(arr7,arr8,arr9,arr10)
            j//=2
        k*=2
    print("final_res: ",arr)
arr = [1,5,3,7]
sort_simu(arr)
print(arr)
