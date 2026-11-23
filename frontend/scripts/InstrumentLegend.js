// ==========================================================
// InstrumentLegend
// - 채보 결과에 나온 악기 목록을 색 + 체크박스로 보여 주는 class
// - 체크를 끄면 그 악기의 음표를 악보에서 숨김 (onToggle로 ScoreRenderer에 전달)
// ==========================================================

export class InstrumentLegend {
	constructor(container, onToggle) {
		this.container = container;
		this.onToggle = onToggle; // (key, visible) => {}
	}

	// instruments: [{ key, name, color, count }]
	render(instruments) {
		this.container.innerHTML = "";
		if (instruments.length === 0) {
			this.container.classList.add("is-hidden");
			return;
		}

		const title = document.createElement("span");
		title.className = "instrument-title text-sub";
		title.textContent = "악기:";
		this.container.appendChild(title);

		instruments.forEach((instrument) => {
			// <label><input type="checkbox"> ■ 피아노 (120)</label>
			const label = document.createElement("label");
			label.className = "instrument-item";

			const checkbox = document.createElement("input");
			checkbox.type = "checkbox";
			checkbox.checked = true;
			checkbox.addEventListener("change", () => {
				this.onToggle?.(instrument.key, checkbox.checked);
			});

			const box = document.createElement("span");
			box.className = "legend-box";
			box.style.backgroundColor = instrument.color;

			const text = document.createElement("span");
			text.textContent = `${instrument.name} (${instrument.count})`;

			label.append(checkbox, box, text);
			this.container.appendChild(label);
		});

		this.container.classList.remove("is-hidden");
	}

	clear() {
		this.container.innerHTML = "";
		this.container.classList.add("is-hidden");
	}
}
