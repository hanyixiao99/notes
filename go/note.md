<!--搜索查询关键词-->



```go
// 声明整形数组
var array [5]int

array := [5]int{1, 2, 3, 4, 5}

array := [...]int{1, 2, 3, 4, 5}

array := [5]int{1:10, 2:20} // [0, 10, 20, 0, 0]

// 声明包含5个元素的指向整数的数组
// 用整形指针初始化索引为0和1的数组元素
array := [5]*int{0: new(int), 1: new(int)}

// 为索引为0和1的元素赋值
*array[0] = 10
*array[1] = 20
```

```go
// 创建切片
slice := make([]string, 5)
slice := make([]int, 3, 5) // 长度为3， 容量为5

slice := []int{10, 20, 30}

// nil slice
var slice []int

```

```go
// 数组和切片创建时的区别
array := [3]int{1, 2, 3}
slice := []int{1, 2, 3}
```

```go
s1 := []int{1, 2}
s2 := []int{3, 4}
append(s1, s2...) // [1, 2, 3, 4]
append([]int(nil), nums...)
```

```go
// defer

package main

import (
    "io"
    "os"
    "fmt"
)

func main() {
    newfile, error := os.Create("learnGo.txt")
    if error != nil {
        fmt.Println("Error: Could not create file.")
        return
    }
    defer newfile.Close()

    if _, error = io.WriteString(newfile, "Learning Go!"); error != nil {
	    fmt.Println("Error: Could not write to file.")
        return
    }

    newfile.Sync()
}

func double(x int) (result int) {
    defer func() { fmt.Printf("double(%d) = %d\n", x,result) }()
    return x + x
}
_ = double(4)
// Output:
// "double(4) = 8"

func triple(x int) (result int) {
    defer func() { result += x }()
    return double(x)
}
fmt.Println(triple(4)) // "12"
```

```go
// log

log.Print("Hey, I'm a log!") // 2022/08/31 15:22:12 Hey, I'm a log!

log.Fatal("Hey, I'm an error log!")
fmt.Print("Can you see me?")
// 2022/08/31 15:23:58 Hey, I'm an error log!
// exit status 1

log.Panic("Hey, I'm an error log!")
fmt.Print("Can you see me?")
// 2022/08/31 15:25:08 Hey, I'm an error log!
// panic: Hey, I'm an error log!
// 
// goroutine 1 [running]:
// log.Panic({0x14000066f48?, 0x140000021a0?, 0x14000048768?})
// /usr/local/go/src/log/log.go:385 +0x68
// main.main()
// /Users/hanyixiao/Documents/code/Go/src/hello/hello.go:9 +0x48
// exit status 2

log.SetPrefix("main(): ")
log.Print("Hey, I'm a log!")
log.Fatal("Hey, I'm an error log!")
// main(): 2022/08/31 15:26:56 Hey, I'm a log!
// main(): 2022/08/31 15:26:56 Hey, I'm an error log!
// exit status 1
```

```go
// panic
// 由于panic会引起程序的崩溃，因此panic一般用于严重错误，如程序内部的逻辑不一致。
// 对于大部分漏洞，我们应该使用Go提供的错误机制，而不是panic，尽量避免程序的崩溃。
```

```go
// recover
// 如果在deferred函数中调用了内置函数recover，并且定义该defer语句的函数发生了panic异常，recover会使程序从panic中恢复，并返回panic value。
// 导致panic异常的函数不会继续运行，但能正常返回。在未发生panic时调用recover，recover会返回nil。
```

```go
// byte & rune 声明时使用 单引号 '' 
// byte 1 字节 8 位 uint8 ASCII 'a' = 97 '0' = 48 'A' = 65
// rune 4 字节 32 位 uint32 Unicode 'a' = 97 
```

```
\a      响铃
\b      退格
\f      换页
\n      换行
\r      回车
\t      制表符
\v      垂直制表符
\'      单引号（只用在 '\'' 形式的rune符号面值中）
\"      双引号（只用在 "..." 形式的字符串面值中）
\\      反斜杠
```

```go
// 字符串和字节 slice 之间可以相互转换
s := "abc"
b := []byte(s)
s2 := string(b)

// 字符串和数字的转换
// 将一个整数转为字符串，一种方法是用 fmt.Sprintf 返回一个格式化的字符串
// 另一个方法是用 strconv.Itoa (“整数到ASCII”)
x := 123
y := fmt.Sprintf("%d", x)
fmt.Println(y, strconv.Itoa(x)) // "123 123"

// FormatInt和FormatUint函数可以用不同的进制来格式化数字：
fmt.Println(strconv.FormatInt(int64(x), 2)) // "1111011"

// 如果要将一个字符串解析为整数，可以使用strconv包的Atoi或ParseInt函数，
// 还有用于解析无符号整数的ParseUint函数：
x, err := strconv.Atoi("123")             // x is an int
y, err := strconv.ParseInt("123", 10, 64) // base 10, up to 64 bits


```

```go
// 链表
dummyHead := new(ListNode) // dummyHead := &ListNode{}
cursor := dummyHead

return dummyhead.Next

dummy := &ListNode{Next: head}
```

```go
// sort

// sort.slice
func Slice(slice interface{}, less func(i, j int) bool) 

sort.Slice(arrs, func(i, j int) bool {
		return arrs[i].Num > arrs[j].Num
	})

sort.Slice(pairs, func(i, j int) bool {
        return pairs[i][1] < pairs[j][1]
    })
```

```go
// 二进制转整数
ans = ans * 2 + head.Val
head = head.Next

// 十转二统计1
func onesCount(x int) (c int) {
    for ; x > 0; x /= 2 {
        c += x % 2
    }
    return
}
```

```go
// 质数筛
func countPrimes(n int) int {
    isPrime := make([]bool, n)
    cnt := 0
    for i := range isPrime {
        isPrime[i] = true
    }
    for i := 2; i < n; i++ {
        if isPrime[i] {
            cnt++
            for j := 2 * i; j < n; j += i {
                isPrime[j] = false
            }
        }
    }
    return cnt
}
```

```go
// 3 的幂
// 3232 位有符号整数的范围内，最大的 33 的幂为 3^{19} = 11622614673^19=1162261467
return n > 0 && 1162261467 % n == 0
```

```go
// 位运算

// 01 1011 0110
// 11 0001 1101
// 11 1011 1111 OR |
// 01 0001 0100 AND &
// 10 1010 1011 XOR ^

// 左移 <<：左边的二进制位丢弃，右边补0 3(11) << 1 = 6(110)
// 右移 >>：正数左补0，负数左补1，右边丢弃 3 >> 1 = 1
// 无符号左移 <<<：左边的二进制位丢弃，右边补0
// 无符号右移 >>>：忽略符号位，空位都以0补齐

// n & (n - 1) 把 n 的二进制位中的最低位的 1 变为 0 
// 6(110) & (6 - 1) = 4(100)

// 对应二进制位不同的位置取1
// x ^ y

// 异或 ^
// 任何数和 0 做异或运算，结果仍然是原来的数，即 a ^ 0 = a
// a ^ a = 0
// a ^ b ^ a = b ^ a ^ a = b ^ (a ^ a) = b
```

```go
// 统计字符串中字符与空格个数

words := strings.Fields(s)
// 将 s 切分成多个子串
// 如果 s 中只包含空白字符，则返回空列表

space := strings.Count(s, " ") 
```

```go
// 常用函数

// 数字转字符串
strconv.Itoa()

// 排序
sort.Ints(intList)
sort.Float64s(float8List)
sort.Strings(stringList)

sort.Slice(arr, func(i, j int) bool { return arr[i][1] > boxTypes[j][1] })
// 字符串转数组
s := "abcd"
sli := []rune(s)

// 数组合并字符串
data := []string{"a", "b", "c"} // 字符串数组
str := string.Join(data, "")

data1 := []byte{'a', 'b', 'c'} // 注意是 '' 不是 "", 字符数组
srt1 := string(data1[:])
```

```go
// 反转链表
func reverseList(head *ListNode) *ListNode {
    var prev *ListNode
    curr := head
    for curr != nil {
        next := curr.Next
        curr.Next = prev
        prev = curr
        curr = next
    }
    return prev
}
```

