# Homebrew cask for Murmur.
#
# Install from this repository:
#   brew install --cask ./scripts/homebrew/murmur.rb
#
# The `version` and `sha256` lines below are placeholders in git. After a
# successful `scripts/release.sh`, that script rewrites exactly these two lines
# in place — it matches them with sed on `^  version "..."$` and
# `^  sha256 "..."$`, so keep them one per line, two spaces of indent, and no
# trailing comment. `release.sh` also writes dist/Murmur-<version>.dmg.sha256 if
# you would rather paste the digest by hand (`UPDATE_CASK=false` skips the
# rewrite entirely).
#
# The DMG is Developer ID signed and notarized when the release CI has the
# signing secrets; Homebrew will refuse an ad-hoc "internal" build on another
# Mac, which is the intended behaviour.

cask "murmur" do
  version "1.0.0"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"

  url "https://github.com/Mbr0/murmur/releases/download/v#{version}/Murmur-#{version}.dmg",
      verified: "github.com/Mbr0/murmur/"
  name "Murmur"
  desc "Local speech-to-text in the macOS menu bar"
  homepage "https://github.com/Mbr0/murmur"

  livecheck do
    url :url
    strategy :github_latest
  end

  # Apple Silicon runs Voxtral through MLX; Intel Macs run the bundled
  # whisper.cpp server (decision D7). Both need a reasonably current macOS.
  depends_on macos: ">= :sonoma"

  app "Murmur.app"

  zap trash: [
    "~/Library/Application Support/Murmur",
    "~/Library/Caches/com.canopystudio.murmur",
    "~/Library/Preferences/com.canopystudio.murmur.plist",
    "~/Library/Saved Application State/com.canopystudio.murmur.savedState",
  ]
end
