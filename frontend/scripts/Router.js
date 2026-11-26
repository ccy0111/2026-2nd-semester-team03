// ==========================================================
// Router
// - 주소의 # 뒤 부분으로 페이지를 바꾸는 class (새로고침 없이 화면만 전환)
//   #/           → 메인(소개) 페이지
//   #/transcribe → 채보 페이지
// - 페이지를 오가도 채보 결과·모델은 그대로 유지됨 (한 HTML 안에서 보이기/숨기기만 함)
// - 메인 페이지의 "떠오르는" 애니메이션(.reveal)도 여기서 처리
// ==========================================================

const PAGES = {
	"": "home",
	transcribe: "transcribe",
};

export class Router {
	// onChange(page, previous): 페이지가 바뀔 때 (예: 채보 페이지를 떠나면 재생 멈추기)
	constructor({ onChange } = {}) {
		this.onChange = onChange;
		this.page = null;
		this.pages = {
			home: document.querySelector("#page-home"),
			transcribe: document.querySelector("#page-transcribe"),
		};
		this.links = document.querySelectorAll(".nav-link");
	}

	start() {
		window.addEventListener("hashchange", () => this.show());
		this.bindScrollLinks();
		this.setupReveal();
		this.show();
	}

	// 지금 주소에 맞는 페이지 보여 주기
	show() {
		const hash = window.location.hash;
		// "#/..."가 아닌 주소(#features 같은 페이지 안 이동)는 페이지를 바꾸지 않음
		if (hash && !hash.startsWith("#/")) return;
		const page = PAGES[hash.slice(2)] ?? "home";
		if (page === this.page) return;

		const previous = this.page;
		this.page = page;
		Object.entries(this.pages).forEach(([name, element]) => {
			element.classList.toggle("is-hidden", name !== page);
		});
		this.links.forEach((link) => {
			const active = link.dataset.page === page;
			link.classList.toggle("is-active", active);
			if (active) link.setAttribute("aria-current", "page");
			else link.removeAttribute("aria-current");
		});
		window.scrollTo(0, 0);
		this.onChange?.(page, previous);
	}

	// "기능 살펴보기"처럼 같은 페이지 안으로 이동하는 링크: 주소를 바꾸지 않고 부드럽게 스크롤
	bindScrollLinks() {
		document.querySelectorAll("[data-scroll]").forEach((link) => {
			link.addEventListener("click", (event) => {
				const target = document.querySelector(link.getAttribute("href"));
				if (!target) return;
				event.preventDefault();
				target.scrollIntoView({ behavior: "smooth" });
			});
		});
	}

	// 화면에 들어온 .reveal 요소에 is-visible을 붙여 떠오르게 함 (한 번만)
	setupReveal() {
		const items = document.querySelectorAll(".reveal");
		if (!("IntersectionObserver" in window)) {
			items.forEach((item) => item.classList.add("is-visible"));
			return;
		}
		const observer = new IntersectionObserver((entries) => {
			entries.forEach((entry) => {
				if (!entry.isIntersecting) return;
				entry.target.classList.add("is-visible");
				observer.unobserve(entry.target);
			});
		}, { threshold: 0.15 });
		items.forEach((item) => observer.observe(item));
	}
}
