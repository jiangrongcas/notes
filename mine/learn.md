# Git

[Git教程 - 廖雪峰的官方网站](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)

# Virtualbox

1. virtualbox -> 设置 -> 常规 -> 高级 -> 共享粘贴板：双向，拖放：双向。
2. virtualbox -> 设置 -> 显示 -> 不要勾选高清(HiDPI)支持，很卡顿，建议勾选启动3D加速和启用和2D视频加速，有些图形软件可能会用到。
3. 文件共享
    - virtualbox -> 设置 -> 共享文件夹 -> 添加一个文件夹，勾选固定分配和自动挂载 -> 点OK保存设置。
    - 启动windows，Device -> Insert Guest Additions CD Image -> 进入我的电脑 -> 打开Guest Additions CD Image -> 运行VBoxWindowsAdditions-amd64.exe。
    - 重启windows后，共享文件夹会显示在我的电脑的网络位置里。
