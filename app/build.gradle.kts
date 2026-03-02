plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.wying.classifydemo"
    compileSdk {
        version = release(36)
    }

    defaultConfig {
        applicationId = "com.wying.classifydemo"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                // 【核心传参】将解压后的路径传递给 CMakeLists.txt
                arguments(
                    "-DANDROID_STL=c++_shared",
                    "-DPYTORCH_DIR=${layout.buildDirectory.get().asFile.absolutePath}/pytorch_lite"
                )
            }
        }

        ndk {
            abiFilters.add("arm64-v8a")
        }
    }

    ndkVersion = "21.4.7075529"

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildFeatures {
        compose = true
    }
}

// 1. 定义一个专门用于提取 C++ 构建依赖的 Configuration
val extractForNativeBuild by configurations.creating

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)

    implementation("org.pytorch:pytorch_android_lite:2.1.0")
    implementation("org.pytorch:pytorch_android_torchvision_lite:2.1.0")
    extractForNativeBuild("org.pytorch:pytorch_android_lite:2.1.0")

    // CameraX
    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)
    implementation(libs.camera.view)

    // Gson for JSON parsing
    implementation("com.google.code.gson:gson:2.11.0")
}

// 2. 注册解压 Task：将 AAR 里的 headers 和 jni 提取到 build/pytorch_lite 目录下
val extractAARForNativeBuild by tasks.registering {
    doLast {
        configurations.getByName("extractForNativeBuild").files.forEach { file ->
            project.copy {
                from(zipTree(file))
                // 使用现代 Gradle API 获取 build 目录
                into(layout.buildDirectory.dir("pytorch_lite"))
                include("headers/**")
                include("jni/**")
            }
        }
    }
}

// 3. 拦截 C++ 构建相关的原生 Task，确保在编译前先执行解压
tasks.withType<Task>().configureEach {
    if (name.contains("generateJsonModel") || name.contains("ExternalNativeBuild")) {
        dependsOn(extractAARForNativeBuild)
    }
}