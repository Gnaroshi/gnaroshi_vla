import numpy as np
import OpenGL.GL as gl
import robosuite


def check_renderer_backend():
    # 1. 테스트용 환경을 가볍게 생성 (Offscreen Renderer 활성화)
    # 이미 사용 중인 env가 있다면 env.reset() 이후에 아래 print문만 넣으시면 됩니다.
    try:
        env = robosuite.make(
            "Lift",
            robots="Panda",
            has_renderer=False,  # 화면 창 띄우기 X
            has_offscreen_renderer=True,  # 내부 렌더링 O
            use_camera_obs=True,  # 카메라 사용 O
            ignore_done=True,
        )
        env.reset()

        # 2. 강제로 렌더링 한 번 수행 (이 시점에 컨텍스트가 생성됨)
        # depth=False 등으로 가볍게 호출
        env.sim.render(camera_name="frontview", width=100, height=100)

        # 3. OpenGL 정보 쿼리 (핵심!)
        renderer = gl.glGetString(gl.GL_RENDERER).decode("utf-8")
        vendor = gl.glGetString(gl.GL_VENDOR).decode("utf-8")
        version = gl.glGetString(gl.GL_VERSION).decode("utf-8")

        print("\n" + "=" * 40)
        print(f"🔍 [렌더링 백엔드 확인 결과]")
        print(f"▶ Vendor (제조사) : {vendor}")
        print(f"▶ Renderer (장치명): {renderer}")
        print(f"▶ Version (버전)   : {version}")

        # 해석 가이드
        if "NVIDIA" in vendor or "NVIDIA" in renderer:
            print("👉 결론: [EGL/GPU] 사용 중 (성공! 빠름)")
        elif "llvmpipe" in renderer or "softpipe" in renderer or "Mesa" in vendor:
            print("👉 결론: [OSMesa/CPU] 소프트웨어 렌더링 사용 중 (느림)")
        else:
            print("👉 결론: 알 수 없는 렌더러 (기타)")
        print("=" * 40 + "\n")

    except Exception as e:
        print("\n❌ 렌더러 초기화 실패 (아직 백엔드 연결 안됨):", e)


if __name__ == "__main__":
    check_renderer_backend()
