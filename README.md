# OptixInOneWeekend

このリポジトリは [レイトレ Advent Calendar 2021](https://qiita.com/advent-calendar/2021/raytracing) の12月20日担当分 「[NVIDIA OptiXでRay tracing in One Weekend](https://qiita.com/sketchbooks99/items/de98db331f8c8d24628c)」のサンプルコードです。

![result.png](result.png)

動作するにはOptiXをインストール後、`SDK/`以下に本リポジトリを配置し、SDK/CMakeLists.txtのビルドディレクトリ指定部分に`add_subdirectory(optixInOneWeekend)`を追加してください。

# 要件
- NVIDIA OptiX 7.4 to 7.7
  - OptiX 7.3以前のバージョンで動作するには、使用バージョンに合わせてAPIを修正してください。