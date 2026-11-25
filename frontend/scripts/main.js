// 진입 파일: 앱과 페이지 전환(Router)을 만들고 시작만 한다.
import { App } from "./App.js";
import { Router } from "./Router.js";

const app = new App();
const router = new Router({
	// 채보 페이지를 떠나면 소리를 모두 멈춤 (결과는 그대로 남아 있음)
	onChange: (page, previous) => {
		if (previous === "transcribe") app.pauseAll();
	},
});

router.start();
app.init(); // 메인 페이지를 보는 동안에도 모델을 미리 준비 → 채보 페이지에서 바로 시작 가능
