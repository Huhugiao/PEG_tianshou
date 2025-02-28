# 打开并读取文件
def read_file_and_print_length(file_name):
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()
            length = len(content)
            print(f"文件内容的长度为: {length} 个字符")
            print("文件内容如下：")
            print(content)
    except FileNotFoundError:
        print("文件未找到，请检查文件名是否正确！")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")

# 示例用法
file_name = "output.txt"
read_file_and_print_length(file_name)