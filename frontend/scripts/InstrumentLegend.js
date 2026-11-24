// ==========================================================
// InstrumentLegend
// - 채보 결과에 나온 악기 목록을 색 + 체크박스로 보여 주는 class
// - 체크를 끄면 그 악기의 음표를 악보에서 숨김 (onToggle로 ScoreRenderer에 전달)
// - 🔊 버튼을 누르면 "악기 소리로 듣기"에서 그 악기 소리를 끔/켬 (onMute로 NotePlayer에 전달)
// ==========================================================

export class InstrumentLegend {
	constructor(container, onToggle, onMute) {
		this.container = container;
		this.onToggle = onToggle; // (key, visible) => {}
		this.onMute = onMute;     // (key, muted) => {}
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

			// 소리 켜기/끄기 버튼 (label 밖에 둬야 눌러도 체크박스가 바뀌지 않음)
			const muteButton = document.createElement("button");
			muteButton.type = "button";
			muteButton.className = "instrument-mute";
			this.setMuteButton(muteButton, instrument.name, false);
			muteButton.addEventListener("click", () => {
				const muted = muteButton.getAttribute("aria-pressed") !== "true";
				this.setMuteButton(muteButton, instrument.name, muted);
				this.onMute?.(instrument.key, muted);
			});

			const item = document.createElement("span");
			item.className = "instrument-entry";
			item.append(label, muteButton);
			this.container.appendChild(item);
		});

		this.container.classList.remove("is-hidden");
	}

	// 버튼 모양: 🔊(소리 켜짐) / 🔇(소리 꺼짐)
	setMuteButton(button, name, muted) {
		button.textContent = muted ? "🔇" : "🔊";
		button.setAttribute("aria-pressed", String(muted));
		button.classList.toggle("is-muted", muted);
		const title = muted ? `${name} 소리 켜기` : `${name} 소리 끄기`;
		button.title = title;
		button.setAttribute("aria-label", title);
	}

	clear() {
		this.container.innerHTML = "";
		this.container.classList.add("is-hidden");
	}
}
