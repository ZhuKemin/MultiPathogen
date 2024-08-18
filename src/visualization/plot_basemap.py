from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import os

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 读取 geojson 文件
geojson_path_province = '../../data/external/basemap-province.geojson'
geojson_path_city = '../../data/external/basemap-city.geojson'
geojson_path_outline = '../../data/external/outline.geojson'

gdf_province = gpd.read_file(geojson_path_province)
gdf_city = gpd.read_file(geojson_path_city)
gdf_outline = gpd.read_file(geojson_path_outline)

# 指定要特殊显示的城市列表
highlight_cities = ['北京市', '广州市', '武汉市', '西安市', '兰州市', '苏州市', '温州市', '云浮市']

# 创建绘图函数
def plot_basemap(gdf_province, gdf_city, gdf_outline, output_dir='./output'):
    fig, ax = plt.subplots(figsize=(16, 12))
    plt.tight_layout()

    # 设置 Basemap 投影和网格
    m = Basemap(projection='lcc', llcrnrlon=78, llcrnrlat=12.5, urcrnrlon=148, urcrnrlat=53,
                lat_1=20., lat_2=40., lon_0=105., resolution='i', area_thresh=1000., ax=ax)
    m.drawparallels(np.arange(20, 60,  10), labels=[1, 0, 0, 0], color='gray', linewidth=0.6, zorder=0,
                fontsize=24, dashes=[5, 5])
    m.drawmeridians(np.arange(70, 160, 10), labels=[0, 0, 0, 1], color='gray', linewidth=0.6, zorder=0,
                fontsize=24, dashes=[5, 5])

    # 绘制省边界
    for _, row in gdf_province.iterrows():
        geom = row['geometry']
        if geom.geom_type == 'Polygon':
            coords = np.array(geom.exterior.coords)
            x, y = m(coords[:, 0], coords[:, 1])
            poly = MplPolygon(np.c_[x, y], facecolor='none', edgecolor='gray', linewidth=0.6, zorder=11)
            ax.add_patch(poly)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                coords = np.array(poly.exterior.coords)
                x, y = m(coords[:, 0], coords[:, 1])
                mpl_poly = MplPolygon(np.c_[x, y], facecolor='none', edgecolor='gray', linewidth=0.6, zorder=11)
                ax.add_patch(mpl_poly)

    # 绘制城市边界
    for _, row in gdf_city.iterrows():
        geom = row['geometry']
        city_name = row['市']
        facecolor = 'tab:cyan' if city_name in highlight_cities else 'white'
        edgecolor = 'black' if city_name in highlight_cities else 'lightgray'
        linewidth = 0.6 if city_name in highlight_cities else 0.4
        zorder = 14 if city_name in highlight_cities else 10

        if geom.geom_type == 'Polygon':
            coords = np.array(geom.exterior.coords)
            x, y = m(coords[:, 0], coords[:, 1])
            poly = MplPolygon(np.c_[x, y], facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
            ax.add_patch(poly)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                coords = np.array(poly.exterior.coords)
                x, y = m(coords[:, 0], coords[:, 1])
                mpl_poly = MplPolygon(np.c_[x, y], facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
                ax.add_patch(mpl_poly)

    # 绘制外轮廓
    for _, row in gdf_outline.iterrows():
        geom = row['geometry']
        if geom.geom_type == 'Polygon':
            coords = np.array(geom.exterior.coords)
            x, y = m(coords[:, 0], coords[:, 1])
            poly = MplPolygon(np.c_[x, y], facecolor='none', edgecolor='black', linewidth=1.5, zorder=12)
            ax.add_patch(poly)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                coords = np.array(poly.exterior.coords)
                x, y = m(coords[:, 0], coords[:, 1])
                mpl_poly = MplPolygon(np.c_[x, y], facecolor='none', edgecolor='black', linewidth=1.5, zorder=12)
                ax.add_patch(mpl_poly)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 添加嵌入地图
    left, bottom, width, height = 0.77, 0.06, 0.20, 0.25
    ax2 = fig.add_axes([left, bottom, width, height])

    # 使用 geopandas 绘制嵌入地图
    gdf_province.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=0.8)
    gdf_outline.plot(ax=ax2, facecolor='none', edgecolor='black',  linewidth=0.8)

    ax2.set_xlim(106, 122)
    ax2.set_ylim(0, 20)
    ax2.grid(ls="-.", lw=0.2, color='lightgray')

    # ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    plt.subplots_adjust(left=0.03)
    plt.savefig(os.path.join(output_dir, "basemap.png"), dpi=600)
    # plt.tight_layout()
    plt.show()

# 调用绘图函数
plot_basemap(gdf_province, gdf_city, gdf_outline, output_dir='../../reports/figures')
