# OptixInOneWeekend

このリポジトリは [レイトレ Advent Calendar 2021](https://qiita.com/advent-calendar/2021/raytracing) の12月20日担当分 「[NVIDIA OptiXでRay tracing in One Weekend]()」のサンプルコードです。

![result.png](result.png)

動作するにはOptiXをインストール後、`SDK/`以下に本リポジトリを配置し、SDK/CMakeLists.txtのビルドディレクトリ指定部分に`add_subdirectory(optixInOneWeekend)`を追加してください。