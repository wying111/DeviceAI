package com.wying.classifydemo

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * 将 Assets 目录下的文件拷贝到 App 的 internal files 目录
 * @param context 上下文
 * @param assetName assets 文件夹中的文件名
 * @return 拷贝后的 File 对象
 */
fun copyAssetToFile(context: Context, assetName: String): File? {
    // 目标文件路径：/data/user/0/包名/files/文件名
    val targetFile = File(context.filesDir, assetName)

    // 性能优化：如果文件已经存在，通常不需要重复拷贝
    // 如果你的文件会更新，可以增加版本校验逻辑
    if (targetFile.exists()) {
        return targetFile
    }

    try {
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(targetFile).use { outputStream ->
                val buffer = ByteArray(1024 * 4) // 4KB 缓冲区
                var length: Int
                while (inputStream.read(buffer).also { length = it } > 0) {
                    outputStream.write(buffer, 0, length)
                }
                outputStream.flush()
            }
        }
        return targetFile
    } catch (e: IOException) {
        e.printStackTrace()
        return null
    }
}