import { defineConfig } from "cf/config";

// A static site: viewer/ (wrangler.config.ts), no Worker code.
export default defineConfig({
	worker: {
		name: "qwen-steering",
		compatibilityDate: "2026-10-01",
		domains: ["qwen-steering.rcn.sh"],
	},
});
