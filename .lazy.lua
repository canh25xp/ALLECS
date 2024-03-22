return {
	{
		"nvim-neo-tree/neo-tree.nvim",
		opts = {
			filesystem = {
				filtered_items = {
					hide_by_pattern = {
						"*/__pycache__",
					},
				},
			},
		},
	},
}
