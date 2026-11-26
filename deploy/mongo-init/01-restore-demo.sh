#!/bin/bash
# mongo 官方镜像只在数据卷为空的首次启动时执行本目录脚本：导入演示数据（不含个人约束 profiles）。
# 已有数据的卷不会被覆盖。想重置为演示数据：docker compose down -v && docker compose up -d
set -e
ARCHIVE=/docker-entrypoint-initdb.d/demo-data.archive.gz
if [ -f "$ARCHIVE" ]; then
  NS_DB="${MONGO_DB:-advisor}"        # 与 app 的 MONGO_DB 保持一致：改库名时演示数据不能导入旧库
  echo "[advisor] 导入演示数据到 ${NS_DB} ..."
  mongorestore --quiet --gzip --archive="$ARCHIVE" --nsFrom='advisor.*' --nsTo="${NS_DB}.*"
  echo "[advisor] 演示数据导入完成"
fi
