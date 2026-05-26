### SINH VIÊN: NGUYỄN TUẤN ANH - 22028310 - K67I-CSI
### SẢN PHẨM KHÓA LUẬN TỐT NGHIỆP: ĐÁNH GIÁ HIỆU NĂNG CÁC THUẬT TOÁN HỌC TĂNG CƯỜNG ONLINE VÀ OFFLINE TRONG BÀI TOÁN ĐIỀU KHIỂN ROBOT 6 BẬC TỰ DO

## HƯỚNG DẪN CÀI ĐẶT
LƯU Ý: TOÀN BỘ DỰ ÁN ĐƯỢC TRIỂN KHAI TRÊN UBUNTU 24.04, YÊU CẦU TẢI VÀ THỰC HIỆN TRÊN UBUNTU 24.04

*Bước 1* : Tải Miniconda tại link : `https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install`

*Bước 2:* tải và cài đặt ROS2 JAZZY theo link hướng dẫn `https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html`

*Bước 3*: Tải và làm theo hướng dẫn cài đặt tại link: `https://ai.robotis.com/omx/introduction_omx.html` để cài đặt Docker container của robot cũng như môi trường

*Bước 4*: Trong Docker container *physical_ai_tools*,tạo miniconda venv với `python=3.12`. Clone repo về máy ```git clone https://github.com/LaRojaPlamya162/omx_controller/new/main.git```, đồng thời tải bóng (vật thể tương tác) bằng lệnh `gz fuel download -u https://fuel.gazebosim.org/1.0/openrobotics/models/cricket%20ball`, tải toàn bộ thư viện cần thiết bằng `pip install -r requirement.txt`

*Bước 5*: Trong Docker container *open_manipulator*. Tại file *open_manipulator/open_manipulator_bringup/launch/omx_f_gazebo.launch.py*, thêm cài đặt bridge (cầu nối các topic/service từ môi trường mô phỏng Gazebo sang chương trình thực thi ROS2 Python)

```
bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_ros_bridge',  
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock', 
            '/world/empty/model/cricket_ball/pose@geometry_msgs/msg/Pose[gz.msgs.Pose',
            '/world/empty/create@ros_gz_interfaces/srv/SpawnEntity',
            '/world/empty/remove@ros_gz_interfaces/srv/DeleteEntity',
            '/world/empty/pose/info@geometry_msgs/msg/PoseArray[gz.msgs.Pose_V',
        ],
        remappings=[
            (
                '/world/empty/model/cricket_ball/pose',
                '/cricket_ball/pose'
            )
        ],
        output='screen'
    )
```
và bóng (vật thể tương tác)
```
gz_spawn_ball = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-file', os.path.expanduser(
    '~/.gz/fuel/fuel.gazebosim.org/openrobotics/models/cricket%20ball/3/model.sdf'
    ),
            '-name', 'cricket_ball',
            '-x', '0.2',
            '-y', '0.2',
            '-z', '0.0',
        ],
    )
```
Ngoài ra, tải file bóng về bằng lệnh `gz download `giống bước 3

*Bước 6:* tại thư mục ros2_ws của container `physical_ai_tool`, thực hiện lệnh sau
```
colcon build --packages-select omx_controller --symlink-install
```
đề đăng ký package *omx_controller* lên hệ thống ROS2 JAZZY

*BƯỚC 7:* Thực hiện lệnh đăng ký lên system 
```
source /opt/ros/jazzy/setup.bash
```

```
source install/setup.bash
```

*Bước 8:* Các thuật toán được lưu trong các controller ghi tên, muốn thực thi chương trình nào thì thực thi lệnh 
```
ros2 run omx_controller <tên_thuật_toán_viết_tắt>_controller
```
Kết quả được lưu tại trong các log của thư mục *models/<tên_thuật_toán_viết_tắt>*

*Bước 9:* Để thực hiện đánh giá kết quả, chỉnh sửa file *src/omx_controller/omx_controller/main.py* và thực thi lệnh 
```
python -m src.omx_controller.omx_controller.main
```