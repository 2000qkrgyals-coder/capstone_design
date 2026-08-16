import streamlit as st

# 1. 사이드바 페이지 리스트 정의 (자기 자신 app.py는 포함하지 않음)
pg = st.navigation(
    [
        st.Page("pages/01_app_1.py", title="메인 화면", icon="🏠"),
        st.Page("pages/02_app_2.py", title="T2 운영 최적화 수정 시스템", icon="📈"),
        st.Page("pages/03_app_3.py", title="가상 운영 시나리오 & AI 의사결정 지원", icon="⚙️"),
    ]
)

# 2. 내비게이션 실행
# pg.run()은 사이드바에 정의된 페이지들만 관리합니다.
# 만약 사이드바에서 아무것도 선택하지 않았을 때(즉, 홈 화면일 때) 실행할 코드를 여기에 씁니다.

if pg.run() == False:
    # 🌟 여기가 메인 화면(홈) 역할을 합니다!
    st.title("메인 화면에 오신 것을 환영합니다! 🏠")
    st.write("사이드바에서 원하는 분석/시뮬레이션 페이지를 선택하세요.")
