#include <jni.h>
#include <string>
#include <android/log.h>
#include <android/bitmap.h>

// 引入 PyTorch Mobile C++ 核心 API
#include <torch/script.h>

#include <torch/csrc/jit/mobile/import.h> // 提供 _load_for_mobile 函数声明
#include <torch/csrc/jit/mobile/module.h> // 提供 torch::jit::mobile::Module 类的完整定义

#define TAG "AI_INFRA"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern "C"
JNIEXPORT jlong JNICALL
Java_com_wying_classifydemo_NativePredictor_loadModel(JNIEnv *env, jobject thiz, jstring model_path) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);

    try {
        // 使用 PyTorch Lite 解释器加载 .ptl 文件
        // 这里在堆内存 (Heap) 上分配了一个 Module 对象
        auto module = new torch::jit::mobile::Module(torch::jit::_load_for_mobile(path));
        env->ReleaseStringUTFChars(model_path, path);

        // 将 C++ 指针强转为 Java 可保存的 jlong (64位整型)
        return reinterpret_cast<jlong>(module);
    } catch (const c10::Error& e) {
        LOGE("加载模型失败: %s", e.what());
        return 0;
    }
}
extern "C"
JNIEXPORT jobject JNICALL
Java_com_wying_classifydemo_NativePredictor_predictImage(JNIEnv *env, jobject thiz, jlong model_ptr, jobject bitmap) {
    // 1. 恢复 C++ 模型对象
    auto* module = reinterpret_cast<torch::jit::mobile::Module*>(model_ptr);
    if (!module) return nullptr;

    // 2. 锁定 Android Bitmap 内存
    AndroidBitmapInfo info;
    void* pixels = nullptr;
    AndroidBitmap_getInfo(env, bitmap, &info);
    AndroidBitmap_lockPixels(env, bitmap, &pixels);

    // 【Infra 核心考点 1：Zero-Copy (零拷贝) 张量创建】
    // torch::from_blob 不会复制内存！它直接让 Tensor 指向 Android Bitmap 的底层像素数组。
    // 这对于性能至关重要，避免了昂贵的内存拷贝开销。
    // Android 的 ARGB_8888 结构对应：高度(H) x 宽度(W) x 4通道(C)
    at::Tensor tensor = torch::from_blob(pixels, {info.height, info.width, 4}, at::kByte);

    // 【Infra 核心考点 2：内存排布转换 (HWC -> NCHW)】
    // 这是端侧 AI 耗时的大头，也是 Infra 工程师必须优化的重点。
    // 手机图像是 HWC 排布（像素点挨个存 RGBA），而神经网络通常要求 NCHW（通道分离存 RRR...GGG...BBB）。

    // a. 去掉 Alpha 通道，只保留 RGB (切片操作)
    tensor = tensor.slice(2, 0, 3); // 维度 2 (通道维度), 从 index 0 取到 3

    // b. 内存重排：HWC (0,1,2) 转换为 CHW (2,0,1)
    tensor = tensor.permute({2, 0, 1});

    // c. 类型转换与归一化：将 0~255 的 Byte 转换为 0.0~1.0 的 Float32
    tensor = tensor.toType(at::kFloat).div(255.0);

    // ================== 【新增：ImageNet 标准化】 ==================
    // 此时 tensor 的形状是 CHW (3, Height, Width)
    // tensor[0] 就是 R 通道，tensor[1] 是 G 通道，tensor[2] 是 B 通道
    // 使用带有下划线的 sub_() 和 div_() 进行 "In-place (原地)" 修改，性能极高，且不会触发 Linker 报错
    tensor[0].sub_(0.485f).div_(0.229f);
    tensor[1].sub_(0.456f).div_(0.224f);
    tensor[2].sub_(0.406f).div_(0.225f);
    // ==============================================================

    // d. 增加 Batch 维度：CHW 变成 1xCxHxW (NCHW)
    tensor = tensor.unsqueeze(0);

    // 由于前面的操作可能会破坏内存连续性，必须调用 contiguous() 确保内存连续，否则推理可能崩溃
    tensor = tensor.contiguous();

    // 3. 执行前向传播 (Forward) 推理
    // 这里底层会调用 NNPACK 或 XNNPACK (基于 ARM NEON 指令集的加速库) 进行计算
    c10::IValue output_val = module->forward({tensor});

    at::Tensor output_tensor = output_val.toTensor().softmax(1);

    // 4. 【修改 2：使用 topk 替代 argmax】
    // 参数含义：k=5(前5名), dim=1(沿着类别维度), largest=true(从大到小降序), sorted=true(保证排好序)
    std::tuple<at::Tensor, at::Tensor> top5_result = output_tensor.topk(5, 1, true, true);

    // std::get<0> 是概率值张量，std::get<1> 是索引张量
    at::Tensor top5_scores = std::get<0>(top5_result).contiguous();
    at::Tensor top5_indices = std::get<1>(top5_result).contiguous();

    // 将张量转换为 C++ 原生指针以便读取
    float* scores_ptr = top5_scores.data_ptr<float>();
    // ⚠️ 避坑警告：PyTorch 的 topk 默认返回的索引是 64 位长整型 (int64_t)，千万不能用 int 接收，否则指针偏移会全错！
    int64_t* indices_ptr = top5_indices.data_ptr<int64_t>();

//    at::Tensor output_tensor = output_val.toTensor(); // 结果是一个 1x1000 的概率张量
//    // 4. 后处理：找到概率最大的类别索引 (Argmax)
//    // 在维度 1 (类别维度) 上求最大值的索引
//    int max_index = output_tensor.argmax(1).item<int>();

    // 5. 释放 Bitmap 内存锁
    AndroidBitmap_unlockPixels(env, bitmap);

    // ==========================================================
    // 6. 【在 C++ 中构建 Java HashMap】
    // ==========================================================

    // a. 找到 Java 的 HashMap 类，并获取它的构造函数和 put 方法
    jclass mapClass = env->FindClass("java/util/HashMap");
    jmethodID mapInit = env->GetMethodID(mapClass, "<init>", "()V");
    jmethodID mapPut = env->GetMethodID(mapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");

    // b. new 一个 HashMap 对象
    jobject hashMap = env->NewObject(mapClass, mapInit);

    // c. 准备 Integer 和 Float 的装箱 (Boxing) 方法
    jclass integerClass = env->FindClass("java/lang/Integer");
    jmethodID intValueOf = env->GetStaticMethodID(integerClass, "valueOf", "(I)Ljava/lang/Integer;");

    jclass floatClass = env->FindClass("java/lang/Float");
    jmethodID floatValueOf = env->GetStaticMethodID(floatClass, "valueOf", "(F)Ljava/lang/Float;");

    // d. 循环 5 次，把结果塞进 HashMap
    for (int i = 0; i < 5; i++) {
        // 将 C++ 的基本类型转换成 Java 的 Integer 和 Float 对象
        jobject keyObj = env->CallStaticObjectMethod(integerClass, intValueOf, (jint)indices_ptr[i]);
        jobject valObj = env->CallStaticObjectMethod(floatClass, floatValueOf, (jfloat)scores_ptr[i]);

        // 调用 hashMap.put(key, value)
        env->CallObjectMethod(hashMap, mapPut, keyObj, valObj);

        // ⚠️ Infra 必修课：释放 JNI 局部引用，防止内存泄漏！
        // 在 for 循环中创建的 JNI 对象，用完必须 DeleteLocalRef，否则循环次数多了会让 JNI 引用表爆满导致 Crash
        env->DeleteLocalRef(keyObj);
        env->DeleteLocalRef(valObj);
    }

    return hashMap; // 把填充好的 Map 返回给 Java/Kotlin
}